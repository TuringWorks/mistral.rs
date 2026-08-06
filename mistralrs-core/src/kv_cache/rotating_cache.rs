use std::sync::Arc;

use candle_core::{Result, Tensor};

use super::codec::{KvCacheCodec, KvCacheCodecRef};
use super::NormalCache;

#[derive(Debug, Clone)]
pub struct RotatingCacheSnapshot {
    pub dim: usize,
    pub current_seq_len: usize,
    pub max_seq_len: usize,
    pub capacity_seq_len: usize,
    /// Retained window in *plain* (decoded) form — `snapshot()` goes through
    /// `current_data()`, which decodes. `restore_from_snapshot` re-encodes on
    /// the way back in, so the codec must travel with the snapshot.
    pub retained: Option<Tensor>,
    pub codec: KvCacheCodecRef,
}

#[derive(Debug, Clone)]
pub struct RotatingCache {
    pub all_data: Option<Tensor>,
    pub dim: usize,
    // The total size of the sequence seen so far.
    pub current_seq_len: usize,
    // max_seq_len is the number of retained tokens in the sliding window.
    pub max_seq_len: usize,
    pub capacity_seq_len: usize,
    // Buffer index one past the newest token; the retained window ends here and slides
    // forward through slack capacity so appends avoid shifting the window every token.
    pub write_pos: usize,
    // The full K/V tensor returned by the last `append()` call.
    // During prefill this may be larger than the internal buffer (retained + new),
    // which is what shared KV layers need for correct attention.
    pub last_append_result: Option<Tensor>,
    // Optional encode/decode hook. `None` is the bit-exact default; `Some`
    // installs a quantization codec (e.g. fp8, TurboQuant). See
    // `super::codec::KvCacheCodec` for the shape/dtype contract.
    //
    // Invariant: `all_data` always holds *encoded* values. Plain tensors are
    // encoded on the way in; buffer slices are decoded on the way out.
    // Buffer-internal relocations move encoded bytes and never touch the codec.
    pub codec: KvCacheCodecRef,
}

impl RotatingCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len: capacity_seq_len.min(max_seq_len),
            write_pos: 0,
            last_append_result: None,
            codec: None,
        }
    }

    /// Install a compression codec. Call before the first `append()` — see
    /// `SingleCache::set_codec` for the rationale.
    pub fn set_codec(&mut self, codec: Arc<dyn KvCacheCodec>) {
        self.codec = Some(codec);
    }

    /// Decode a slice read out of `all_data`. No-op when no codec is installed.
    fn decode(&self, view: Tensor) -> Result<Tensor> {
        match &self.codec {
            Some(codec) => codec.decode(&view),
            None => Ok(view),
        }
    }

    /// Encode a plain tensor on its way into `all_data`. No-op when no codec
    /// is installed.
    fn encode(&self, src: &Tensor) -> Result<Tensor> {
        match &self.codec {
            Some(codec) => codec.encode(src),
            None => Ok(src.clone()),
        }
    }

    fn retained_len(&self) -> usize {
        self.current_seq_len.min(self.max_seq_len)
    }

    fn window_start(&self) -> usize {
        self.write_pos - self.retained_len()
    }

    // Slack beyond the window bounds how often the window is relocated to the front.
    fn max_capacity(&self) -> usize {
        self.max_seq_len + NormalCache::CACHE_GROW_SIZE
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> Option<&Tensor> {
        self.all_data.as_ref()
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => Some(
                self.decode(d.narrow(self.dim, self.window_start(), self.retained_len())?)?,
            ),
        };
        Ok(data)
    }

    pub fn last_append_result(&self) -> Option<&Tensor> {
        self.last_append_result.as_ref()
    }

    pub fn snapshot(&self) -> Result<RotatingCacheSnapshot> {
        Ok(RotatingCacheSnapshot {
            dim: self.dim,
            current_seq_len: self.current_seq_len,
            max_seq_len: self.max_seq_len,
            capacity_seq_len: self.capacity_seq_len,
            retained: self.current_data()?,
            codec: self.codec.clone(),
        })
    }

    pub fn can_append_from_snapshot(
        &self,
        snapshot: &RotatingCacheSnapshot,
        append_len: usize,
    ) -> bool {
        snapshot.current_seq_len == self.current_seq_len
            && snapshot.max_seq_len == self.max_seq_len
            && append_len <= self.max_seq_len
    }

    pub fn accepted_append_from_batched_append(
        &self,
        snapshot: &RotatingCacheSnapshot,
        keep_len: usize,
        row_idx: usize,
        batch_len: usize,
    ) -> Result<Option<Tensor>> {
        let accepted_len = keep_len
            .checked_sub(snapshot.current_seq_len)
            .ok_or_else(|| {
                candle_core::Error::Msg("rotating cache rollback keep_len underflow".into())
            })?;
        if accepted_len == 0 {
            return Ok(None);
        }
        let appended = self.last_append_result.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("missing rotating cache append result".into())
        })?;
        let dim0 = appended.dim(0)?;
        if batch_len == 0 || dim0 % batch_len != 0 {
            candle_core::bail!(
                "rotating cache batch shape mismatch: dim0={dim0}, batch_len={batch_len}"
            );
        }
        let per_row = dim0 / batch_len;
        let retained_len = snapshot.current_seq_len.min(snapshot.max_seq_len);
        appended
            .narrow(0, row_idx * per_row, per_row)?
            .narrow(snapshot.dim, retained_len, accepted_len)?
            .contiguous()
            .map(Some)
    }

    pub fn restore_from_snapshot(
        snapshot: &RotatingCacheSnapshot,
        accepted_append: Option<Tensor>,
        keep_len: usize,
    ) -> Result<Self> {
        let accepted_len = keep_len
            .checked_sub(snapshot.current_seq_len)
            .ok_or_else(|| {
                candle_core::Error::Msg("rotating cache rollback keep_len underflow".into())
            })?;
        if let Some(accepted_append) = accepted_append.as_ref() {
            if accepted_append.dim(snapshot.dim)? != accepted_len {
                candle_core::bail!(
                    "rotating cache rollback accepted append length mismatch: got {}, expected {accepted_len}",
                    accepted_append.dim(snapshot.dim)?
                );
            }
        } else if accepted_len != 0 {
            candle_core::bail!(
                "rotating cache rollback missing accepted append for accepted_len={accepted_len}"
            );
        }

        let retained = match (snapshot.retained.as_ref(), accepted_append.as_ref()) {
            (Some(retained), Some(accepted)) => Tensor::cat(&[retained, accepted], snapshot.dim)?,
            (Some(retained), None) => retained.clone(),
            (None, Some(accepted)) => accepted.clone(),
            (None, None) => {
                return Ok(Self {
                    all_data: None,
                    dim: snapshot.dim,
                    current_seq_len: keep_len,
                    max_seq_len: snapshot.max_seq_len,
                    capacity_seq_len: snapshot.capacity_seq_len.min(snapshot.max_seq_len),
                    write_pos: 0,
                    last_append_result: None,
                    codec: snapshot.codec.clone(),
                });
            }
        };

        let retained_len = retained.dim(snapshot.dim)?;
        let keep = retained_len.min(snapshot.max_seq_len);
        let retained = retained
            .narrow(snapshot.dim, retained_len - keep, keep)?
            .contiguous()?;
        let capacity_seq_len = snapshot
            .capacity_seq_len
            .max(keep)
            .min(snapshot.max_seq_len)
            .max(keep);
        let mut shape = retained.dims().to_vec();
        shape[snapshot.dim] = capacity_seq_len;
        let all_data = Tensor::zeros(shape, retained.dtype(), retained.device())?;
        if keep > 0 {
            // `retained` is plain here: `snapshot.retained` came from
            // `current_data()` (decoded) and `accepted_append` came from
            // `last_append_result` (also decoded). Re-encode so the restored
            // buffer keeps the "all_data is encoded" invariant. For a lossy
            // codec this is one extra encode of already-quantized values,
            // which lands on the same reconstruction grid.
            let to_store = match &snapshot.codec {
                Some(codec) => codec.encode(&retained)?,
                None => retained,
            };
            all_data.slice_set(&to_store, snapshot.dim, 0)?;
        }

        Ok(Self {
            all_data: Some(all_data),
            dim: snapshot.dim,
            current_seq_len: keep_len,
            max_seq_len: snapshot.max_seq_len,
            capacity_seq_len,
            write_pos: keep,
            last_append_result: None,
            codec: snapshot.codec.clone(),
        })
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
        self.write_pos = 0;
        self.last_append_result = None;
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        if len > self.current_seq_len {
            candle_core::bail!(
                "Sliding KV cache cannot extend via set_len (current {}, requested {})",
                self.current_seq_len,
                len,
            );
        }
        // Once the retained window has dropped old tokens, rollback would require
        // data that is no longer present.
        if self.current_seq_len > self.max_seq_len && len < self.current_seq_len {
            candle_core::bail!(
                "Sliding KV cache cannot roll back after truncation \
                 (current_seq_len {} > max_seq_len {}, requested len {})",
                self.current_seq_len,
                self.max_seq_len,
                len,
            );
        }
        if self.current_seq_len.saturating_sub(len) > self.max_seq_len {
            candle_core::bail!(
                "Sliding KV cache tried to reset to len {len} while current is {} and max retained is {}",
                self.current_seq_len,
                self.max_seq_len
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        self.try_set_len(len)?;
        if len < self.current_seq_len {
            self.write_pos -= self.current_seq_len - len;
        }
        self.current_seq_len = len;
        self.last_append_result = None;
        Ok(())
    }

    pub fn append(&mut self, src: &Tensor) -> Result<Tensor> {
        let seq_len = src.dim(self.dim)?;
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            self.all_data = Some(Tensor::zeros(shape, src.dtype(), src.device())?);
        }

        let retained_len = self.retained_len();
        let window_start = self.window_start();

        // During prefill (seq_len > 1), if total tokens exceed the sliding window,
        // we need the full K/V (retained + new) for correct attention: different
        // query positions attend to different windows. Read retained BEFORE the
        // buffer is relocated or overwritten below.
        //
        // Codec note: the buffer holds encoded values; `src` is plain. This
        // tensor is consumed by attention, so decode the retained slice before
        // concatenating it with plain `src`.
        let prefill_full_kv = if seq_len > 1 && (retained_len + seq_len) > self.max_seq_len {
            let ad = self.all_data.as_ref().unwrap();
            Some(if retained_len > 0 {
                let retained = self.decode(
                    ad.narrow(self.dim, window_start, retained_len)?
                        .contiguous()?,
                )?;
                Tensor::cat(&[&retained, &src.contiguous()?], self.dim)?
            } else {
                src.clone()
            })
        } else {
            None
        };

        if seq_len >= self.max_seq_len {
            if self.capacity_seq_len < self.max_seq_len {
                self.capacity_seq_len = self.max_seq_len;
                let mut shape = src.dims().to_vec();
                shape[self.dim] = self.capacity_seq_len;
                self.all_data = Some(Tensor::zeros(shape, src.dtype(), src.device())?);
            }
            // `src` is plain — encode before it lands in the buffer.
            let to_copy = self.encode(
                &src.narrow(self.dim, seq_len - self.max_seq_len, self.max_seq_len)?
                    .contiguous()?,
            )?;
            let ad = self.all_data.as_mut().unwrap();
            ad.slice_set(&to_copy, self.dim, 0)?;
            self.write_pos = self.max_seq_len;
        } else {
            if self.write_pos + seq_len > self.capacity_seq_len {
                let keep_len = retained_len.min(self.max_seq_len - seq_len);
                let keep_start = window_start + retained_len - keep_len;
                let max_capacity = self.max_capacity();
                // Everything below relocates the retained window *within* the
                // buffer (or into a freshly grown one). Those bytes are already
                // encoded, so the codec is deliberately not applied here —
                // running it would double-quantize the retained window.
                if self.capacity_seq_len < max_capacity {
                    let needed = keep_len + seq_len;
                    let n_blocks = needed
                        .div_ceil(NormalCache::CACHE_GROW_SIZE)
                        .max(self.capacity_seq_len / NormalCache::CACHE_GROW_SIZE + 1);
                    self.capacity_seq_len =
                        (n_blocks * NormalCache::CACHE_GROW_SIZE).min(max_capacity);
                    let mut shape = src.dims().to_vec();
                    shape[self.dim] = self.capacity_seq_len;
                    let old = self.all_data.take().unwrap();
                    let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
                    if keep_len > 0 {
                        let retained = old.narrow(self.dim, keep_start, keep_len)?.contiguous()?;
                        ad.slice_set(&retained, self.dim, 0)?;
                    }
                    self.all_data = Some(ad);
                } else if keep_len > 0 {
                    let ad = self.all_data.as_mut().unwrap();
                    let retained = ad
                        .narrow(self.dim, keep_start, keep_len)?
                        .copy()?
                        .contiguous()?;
                    ad.slice_set(&retained, self.dim, 0)?;
                }
                self.write_pos = keep_len;
            }
            // Encode before taking the mutable borrow of `all_data`.
            let to_store = self.encode(&src.contiguous()?)?;
            let write_pos = self.write_pos;
            let ad = self.all_data.as_mut().unwrap();
            ad.slice_set(&to_store, self.dim, write_pos)?;
            self.write_pos += seq_len;
        }

        self.current_seq_len += seq_len;

        let result = if let Some(full_kv) = prefill_full_kv {
            full_kv
        } else {
            let ad = self.all_data.as_ref().unwrap();
            let view = ad.narrow(self.dim, self.window_start(), self.retained_len())?;
            self.decode(view)?
        };

        self.last_append_result = Some(result.clone());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::{Device, Tensor};

    use super::super::codec::PassthroughCodec;
    use super::RotatingCache;

    fn make_src(values: &[f32]) -> candle_core::Result<Tensor> {
        Tensor::new(values.to_vec(), &Device::Cpu)?.reshape((1, 1, values.len(), 1))
    }

    fn make_batched_src(rows: &[&[f32]]) -> candle_core::Result<Tensor> {
        let len = rows.first().map(|row| row.len()).unwrap_or(0);
        assert!(rows.iter().all(|row| row.len() == len));
        let values = rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        Tensor::new(values, &Device::Cpu)?.reshape((rows.len(), 1, len, 1))
    }

    #[test]
    fn retains_last_window_in_order() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 4, 4);

        let first = cache.append(&make_src(&[0., 1., 2.])?)?;
        assert_eq!(first.flatten_all()?.to_vec1::<f32>()?, vec![0., 1., 2.]);
        assert_eq!(cache.current_seq_len(), 3);

        let second = cache.append(&make_src(&[3., 4., 5.])?)?;
        // During multi-token append (prefill), full K/V is returned so all
        // query positions can attend to their correct sliding windows.
        assert_eq!(
            second.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 1., 2., 3., 4., 5.]
        );
        assert_eq!(cache.current_seq_len(), 6);

        let current = cache.current_data()?.unwrap();
        assert_eq!(
            current.flatten_all()?.to_vec1::<f32>()?,
            vec![2., 3., 4., 5.]
        );

        Ok(())
    }

    #[test]
    fn rejects_rollback_after_truncation() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 4, 4);
        let _ = cache.append(&make_src(&[0., 1., 2., 3., 4.])?)?;

        assert!(cache.try_set_len(4).is_err());
        assert!(cache.set_len(4).is_err());

        Ok(())
    }

    #[test]
    fn restores_from_snapshot_after_sliding_window_advance() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 4, 4);
        let _ = cache.append(&make_src(&[0., 1., 2., 3., 4.])?)?;
        let snapshot = cache.snapshot()?;

        let _ = cache.append(&make_src(&[5., 6., 7.])?)?;
        let accepted = cache.accepted_append_from_batched_append(&snapshot, 7, 0, 1)?;
        assert_eq!(
            accepted.as_ref().unwrap().flatten_all()?.to_vec1::<f32>()?,
            vec![5., 6.]
        );

        let restored = RotatingCache::restore_from_snapshot(&snapshot, accepted, 7)?;
        assert_eq!(restored.current_seq_len(), 7);
        assert_eq!(
            restored
                .current_data()?
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![3., 4., 5., 6.]
        );

        Ok(())
    }

    #[test]
    fn extracts_accepted_append_from_batched_append_row() -> candle_core::Result<()> {
        let mut single = RotatingCache::new(2, 4, 4);
        let _ = single.append(&make_src(&[0., 1., 2.])?)?;
        let snapshot = single.snapshot()?;

        let mut batched = RotatingCache::new(2, 4, 4);
        let _ = batched.append(&make_batched_src(&[&[0., 1., 2.], &[10., 11., 12.]])?)?;
        let _ = batched.append(&make_batched_src(&[&[3., 4., 5.], &[13., 14., 15.]])?)?;

        let accepted = batched.accepted_append_from_batched_append(&snapshot, 5, 1, 2)?;
        assert_eq!(
            accepted.unwrap().flatten_all()?.to_vec1::<f32>()?,
            vec![13., 14.]
        );

        Ok(())
    }

    #[test]
    fn returns_full_kv_on_large_prefill() -> candle_core::Result<()> {
        // Sliding window = 4, but prefill has 7 tokens
        let mut cache = RotatingCache::new(2, 4, 4);

        // Prefill with more tokens than the window
        let result = cache.append(&make_src(&[0., 1., 2., 3., 4., 5., 6.])?)?;
        // Should return ALL 7 tokens for correct attention during prefill
        assert_eq!(
            result.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 1., 2., 3., 4., 5., 6.]
        );
        assert_eq!(cache.current_seq_len(), 7);

        // Internal buffer should only retain the last 4
        let current = cache.current_data()?.unwrap();
        assert_eq!(
            current.flatten_all()?.to_vec1::<f32>()?,
            vec![3., 4., 5., 6.]
        );

        // Subsequent decode (single token) should work normally
        let decode = cache.append(&make_src(&[7.])?)?;
        assert_eq!(
            decode.flatten_all()?.to_vec1::<f32>()?,
            vec![4., 5., 6., 7.]
        );

        Ok(())
    }

    #[test]
    fn returns_full_kv_on_prefill_with_retained() -> candle_core::Result<()> {
        // Sliding window = 4, initial small append, then large prefill
        let mut cache = RotatingCache::new(2, 4, 4);

        // First: small append (fits in window)
        let first = cache.append(&make_src(&[0., 1., 2.])?)?;
        assert_eq!(first.flatten_all()?.to_vec1::<f32>()?, vec![0., 1., 2.]);

        // Second: prefill that overflows window (retained=3 + new=3 = 6 > 4)
        let second = cache.append(&make_src(&[3., 4., 5.])?)?;
        // Should return retained + new = all 6 tokens
        assert_eq!(
            second.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 1., 2., 3., 4., 5.]
        );

        // Internal buffer should only retain the last 4
        let current = cache.current_data()?.unwrap();
        assert_eq!(
            current.flatten_all()?.to_vec1::<f32>()?,
            vec![2., 3., 4., 5.]
        );

        Ok(())
    }

    #[test]
    fn grows_small_initial_capacity_for_large_first_prefill() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 1024, 512);
        let values = (0_u16..1024).map(f32::from).collect::<Vec<_>>();

        let result = cache.append(&make_src(&values)?)?;

        assert_eq!(result.flatten_all()?.to_vec1::<f32>()?, values);
        assert_eq!(cache.capacity_seq_len, 1024);
        Ok(())
    }

    #[test]
    fn compacts_full_window_before_large_append() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 1024, 512);
        let first = (0_u16..512).map(f32::from).collect::<Vec<_>>();
        let second = (512_u16..1024).map(f32::from).collect::<Vec<_>>();
        let third = (1024_u16..1624).map(f32::from).collect::<Vec<_>>();

        let _ = cache.append(&make_src(&first)?)?;
        let _ = cache.append(&make_src(&second)?)?;
        let result = cache.append(&make_src(&third)?)?;

        let expected_result = (0_u16..1624).map(f32::from).collect::<Vec<_>>();
        assert_eq!(result.flatten_all()?.to_vec1::<f32>()?, expected_result);
        let expected_retained = (600_u16..1624).map(f32::from).collect::<Vec<_>>();
        assert_eq!(
            cache
                .current_data()?
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?,
            expected_retained
        );
        Ok(())
    }

    /// Installing a PassthroughCodec must not change observed values. Covers
    /// the prefill path that decodes the retained window and concatenates it
    /// with plain `src`.
    #[test]
    fn passthrough_codec_roundtrip() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 4, 4);
        cache.set_codec(Arc::new(PassthroughCodec));

        let first = cache.append(&make_src(&[0., 1., 2.])?)?;
        assert_eq!(first.flatten_all()?.to_vec1::<f32>()?, vec![0., 1., 2.]);

        // Prefill that overflows the sliding window — exercises the
        // decode-retained + concat-with-src code path.
        let second = cache.append(&make_src(&[3., 4., 5.])?)?;
        assert_eq!(
            second.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 1., 2., 3., 4., 5.]
        );

        let current = cache.current_data()?.unwrap();
        assert_eq!(
            current.flatten_all()?.to_vec1::<f32>()?,
            vec![2., 3., 4., 5.]
        );

        Ok(())
    }

    /// The codec must survive the ring-buffer relocation that happens when
    /// `write_pos` runs past `capacity_seq_len`, and single-token decode
    /// appends must still read back in order. This is the path the upstream
    /// v0.9 rewrite introduced — the old cache never relocated.
    #[test]
    fn passthrough_codec_survives_window_relocation() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 4, 4);
        cache.set_codec(Arc::new(PassthroughCodec));

        for i in 0..12 {
            let _ = cache.append(&make_src(&[i as f32])?)?;
        }

        assert!(cache.codec.is_some(), "codec dropped during relocation");
        assert_eq!(cache.current_seq_len(), 12);
        assert_eq!(
            cache
                .current_data()?
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![8., 9., 10., 11.]
        );

        Ok(())
    }

    /// snapshot/restore round-trips plain values and carries the codec across,
    /// so a restored cache keeps compressing subsequent appends.
    #[test]
    fn snapshot_restore_preserves_codec() -> candle_core::Result<()> {
        let mut cache = RotatingCache::new(2, 4, 4);
        cache.set_codec(Arc::new(PassthroughCodec));
        let _ = cache.append(&make_src(&[0., 1., 2., 3., 4.])?)?;

        let snapshot = cache.snapshot()?;
        assert!(snapshot.codec.is_some());

        let _ = cache.append(&make_src(&[5., 6., 7.])?)?;
        let accepted = cache.accepted_append_from_batched_append(&snapshot, 7, 0, 1)?;
        let restored = RotatingCache::restore_from_snapshot(&snapshot, accepted, 7)?;

        assert!(restored.codec.is_some(), "codec lost across restore");
        assert_eq!(restored.current_seq_len(), 7);

        Ok(())
    }
}
