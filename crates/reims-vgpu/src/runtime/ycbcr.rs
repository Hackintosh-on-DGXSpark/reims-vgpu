//! Biplanar YCbCr surfaces, expanded to RGBA8 for sampling.
//!
//! From macOS 14 on, Setup Assistant plays its background video into IOSurfaces
//! with a luma plane and an interleaved CbCr plane (`'420f'` on 14, `'422f'` on
//! 15, `'444f'` and ten-bit `'x420'` on 15 and 27) and binds the whole surface
//! as one texture in one of Metal's private biplanar formats. They sample in two
//! ways, each read off the shader that consumes it:
//!
//! - **500 to 510** convert in the sampler. 14's compositor reads `.rgb` and
//!   goes straight on to its transfer function, so it is handed R'G'B'.
//! - **540 to 543** (8-bit 4:2:0, 4:2:2 and 4:4:4, and ten-bit 4:2:0) return
//!   Cr, Y and Cb in red, green and blue: VideoToolbox's transfer kernels on 15
//!   and 27 swizzle `.gbr` and apply their own matrix. The ten-bit kernel
//!   swizzles when it stores the sample in a local array and reads `.rgb` back
//!   later, which was once misread as the kernel taking Y, Cb and Cr.
//!
//! The rest of the private range has not been met, and is refused rather than
//! guessed.
//!
//! Where the device converts, the matrix is BT.709 for HD and larger and BT.601
//! below that, as CoreVideo assumes for a buffer with no matrix attachment; 15's
//! kernel applies BT.601 to its 512 by 512 frames. The range comes from the
//! surface's FourCC: a trailing `f` is full range, anything else video range.
//!
//! Ten-bit surfaces (`'x420'`, the background video after the language panel)
//! hold each sample in the high bits of 16; the high byte is taken, which is
//! what an RGBA8 result can hold anyway.

/// How a texture in one of these formats samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    /// The sampler converts to R'G'B'.
    Converted,
    /// The sampler returns Cr, Y and Cb in red, green and blue.
    CrYCb,
}

impl Sampling {
    /// The family a Metal pixel format belongs to, or `None` for a format that
    /// is not one of the biplanar families this module serves.
    pub fn of(format: u16) -> Option<Self> {
        match format {
            500..=510 => Some(Self::Converted),
            540..=543 => Some(Self::CrYCb),
            _ => None,
        }
    }
}

/// How one biplanar surface stores its chroma.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Biplanar {
    pub width: u32,
    pub height: u32,
    /// Luma texels per chroma texel, across and down: 2 and 2 for 4:2:0.
    pub chroma_x: u32,
    pub chroma_y: u32,
    pub full_range: bool,
    /// Bytes a luma sample takes: 1, or 2 for a ten-bit surface.
    pub sample_bytes: u32,
}

impl Biplanar {
    /// The layout two plane records describe, or `None` when they are not a
    /// luma plane followed by a CbCr plane of the same sample width (8 or 16
    /// bits) that subsamples it by one or two in each direction.
    pub fn from_planes(
        fourcc: u32,
        luma: (u32, u32, u16),
        chroma: (u32, u32, u16),
    ) -> Option<Self> {
        let (width, height, luma_bpe) = luma;
        let (chroma_width, chroma_height, chroma_bpe) = chroma;
        let sample_bytes = u32::from(luma_bpe);
        if !matches!(sample_bytes, 1 | 2)
            || u32::from(chroma_bpe) != 2 * sample_bytes
            || width == 0
            || height == 0
        {
            return None;
        }
        let factor = |full: u32, sub: u32| match sub {
            0 => None,
            _ if sub == full => Some(1),
            _ if sub == full.div_ceil(2) => Some(2),
            _ => None,
        };
        Some(Self {
            width,
            height,
            chroma_x: factor(width, chroma_width)?,
            chroma_y: factor(height, chroma_height)?,
            full_range: fourcc & 0xff == u32::from(b'f'),
            sample_bytes,
        })
    }

    pub fn chroma_width(self) -> u32 {
        self.width.div_ceil(self.chroma_x)
    }

    pub fn chroma_height(self) -> u32 {
        self.height.div_ceil(self.chroma_y)
    }
}

/// 16.16 fixed-point coefficients: luma scale, then Cr to R, Cb to G, Cr to G
/// and Cb to B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Matrix {
    luma: i32,
    luma_offset: i32,
    cr_r: i32,
    cb_g: i32,
    cr_g: i32,
    cb_b: i32,
}

fn fixed(value: f64) -> i32 {
    (value * 65536.0).round() as i32
}

fn matrix(layout: Biplanar) -> Matrix {
    let hd = layout.height > 576 || layout.width > 1024;
    // Kr and Kb; the rest follows from them.
    let (kr, kb) = if hd { (0.2126, 0.0722) } else { (0.299, 0.114) };
    let kg = 1.0 - kr - kb;
    let (luma, luma_offset, chroma) = if layout.full_range {
        (1.0, 0, 1.0)
    } else {
        (255.0 / 219.0, 16, 255.0 / 224.0)
    };
    Matrix {
        luma: fixed(luma),
        luma_offset,
        cr_r: fixed(2.0 * (1.0 - kr) * chroma),
        cb_g: fixed(2.0 * (1.0 - kb) * kb / kg * chroma),
        cr_g: fixed(2.0 * (1.0 - kr) * kr / kg * chroma),
        cb_b: fixed(2.0 * (1.0 - kb) * chroma),
    }
}

fn clamp8(value: i32) -> u8 {
    ((value + 0x8000) >> 16).clamp(0, 255) as u8
}

/// Expand one surface. `luma` and `chroma` hold the planes' rows at their
/// strides; `out` receives `width * height` RGBA8 texels with alpha 255.
pub fn to_rgba8(
    layout: Biplanar,
    sampling: Sampling,
    luma: &[u8],
    luma_stride: usize,
    chroma: &[u8],
    chroma_stride: usize,
    out: &mut [u8],
) -> bool {
    let (w, h) = (layout.width as usize, layout.height as usize);
    let (cw, ch) = (
        layout.chroma_width() as usize,
        layout.chroma_height() as usize,
    );
    let s = layout.sample_bytes as usize;
    if luma_stride < w * s
        || chroma_stride < cw * 2 * s
        || luma.len() < luma_stride * (h - 1) + w * s
        || chroma.len() < chroma_stride * (ch - 1) + cw * 2 * s
        || out.len() < w * h * 4
    {
        return false;
    }
    let m = matrix(layout);
    for y in 0..h {
        let luma_row = &luma[y * luma_stride..y * luma_stride + w * s];
        let chroma_row_start = (y / layout.chroma_y as usize) * chroma_stride;
        let chroma_row = &chroma[chroma_row_start..chroma_row_start + cw * 2 * s];
        let out_row = &mut out[y * w * 4..(y + 1) * w * 4];
        // The high byte of each sample: the sample itself at 8 bits.
        let luma_texels = luma_row.chunks_exact(s).map(|sample| sample[s - 1]);
        for (x, (luma_texel, texel)) in luma_texels.zip(out_row.chunks_exact_mut(4)).enumerate() {
            let c = (x / layout.chroma_x as usize) * 2 * s;
            let (cb, cr) = (chroma_row[c + s - 1], chroma_row[c + 2 * s - 1]);
            match sampling {
                Sampling::CrYCb => {
                    texel.copy_from_slice(&[cr, luma_texel, cb, 255]);
                    continue;
                }
                Sampling::Converted => {}
            }
            let (cb, cr) = (i32::from(cb) - 128, i32::from(cr) - 128);
            let l = (i32::from(luma_texel) - m.luma_offset) * m.luma;
            texel[0] = clamp8(l + m.cr_r * cr);
            texel[1] = clamp8(l - m.cb_g * cb - m.cr_g * cr);
            texel[2] = clamp8(l + m.cb_b * cb);
            texel[3] = 255;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fourcc(code: &[u8; 4]) -> u32 {
        u32::from_be_bytes(*code)
    }

    #[test]
    fn the_two_families_sample_differently() {
        assert_eq!(Sampling::of(500), Some(Sampling::Converted));
        assert_eq!(Sampling::of(541), Some(Sampling::CrYCb));
        assert_eq!(Sampling::of(542), Some(Sampling::CrYCb));
        assert_eq!(Sampling::of(543), Some(Sampling::CrYCb));
        assert_eq!(Sampling::of(520), None);
        assert_eq!(Sampling::of(544), None);
        assert_eq!(Sampling::of(80), None);
    }

    #[test]
    fn plane_records_name_the_subsampling() {
        let layout =
            Biplanar::from_planes(fourcc(b"420f"), (3840, 2160, 1), (1920, 1080, 2)).unwrap();
        assert_eq!(
            (layout.chroma_x, layout.chroma_y, layout.full_range),
            (2, 2, true)
        );
        let layout = Biplanar::from_planes(fourcc(b"422v"), (512, 512, 1), (256, 512, 2)).unwrap();
        assert_eq!(
            (layout.chroma_x, layout.chroma_y, layout.full_range),
            (2, 1, false)
        );
        let layout = Biplanar::from_planes(fourcc(b"x420"), (64, 64, 2), (32, 32, 4)).unwrap();
        assert_eq!((layout.sample_bytes, layout.full_range), (2, false));
        assert_eq!(
            Biplanar::from_planes(fourcc(b"x420"), (64, 64, 2), (32, 32, 2)),
            None
        );
        assert_eq!(
            Biplanar::from_planes(fourcc(b"420v"), (64, 64, 1), (16, 16, 2)),
            None
        );
    }

    #[test]
    fn video_range_black_white_and_red() {
        let layout = Biplanar::from_planes(fourcc(b"420v"), (2, 2, 1), (1, 1, 2)).unwrap();
        let mut out = [0u8; 16];
        assert!(to_rgba8(
            layout,
            Sampling::Converted,
            &[16, 235, 16, 235],
            2,
            &[128, 128],
            2,
            &mut out
        ));
        assert_eq!(&out[0..4], &[0, 0, 0, 255]);
        assert_eq!(&out[4..8], &[255, 255, 255, 255]);
        // BT.601 red at video range: Y 81, Cb 90, Cr 240.
        assert!(to_rgba8(
            layout,
            Sampling::Converted,
            &[81; 4],
            2,
            &[90, 240],
            2,
            &mut out
        ));
        let red = &out[0..3];
        assert!(red[0] >= 254 && red[1] <= 1 && red[2] <= 1, "{red:?}");
    }

    #[test]
    fn full_range_grey_stays_grey() {
        let layout = Biplanar::from_planes(fourcc(b"422f"), (2, 1, 1), (1, 1, 2)).unwrap();
        let mut out = [0u8; 8];
        assert!(to_rgba8(
            layout,
            Sampling::Converted,
            &[0, 200],
            2,
            &[128, 128],
            2,
            &mut out
        ));
        assert_eq!(out, [0, 0, 0, 255, 200, 200, 200, 255]);
    }

    #[test]
    fn raw_sampling_puts_luma_in_green() {
        let layout = Biplanar::from_planes(fourcc(b"422f"), (2, 1, 1), (1, 1, 2)).unwrap();
        let mut out = [0u8; 8];
        assert!(to_rgba8(
            layout,
            Sampling::CrYCb,
            &[10, 20],
            2,
            &[30, 40],
            2,
            &mut out
        ));
        assert_eq!(out, [40, 10, 30, 255, 40, 20, 30, 255]);
    }

    #[test]
    fn ten_bit_samples_use_their_high_byte() {
        let layout = Biplanar::from_planes(fourcc(b"x422"), (2, 1, 2), (1, 1, 4)).unwrap();
        let mut out = [0u8; 8];
        let luma = [0x00, 0x0a, 0xc0, 0x14];
        let chroma = [0x40, 0x1e, 0x00, 0x28];
        assert!(to_rgba8(
            layout,
            Sampling::CrYCb,
            &luma,
            4,
            &chroma,
            4,
            &mut out
        ));
        assert_eq!(out, [40, 10, 30, 255, 40, 20, 30, 255]);
    }

    #[test]
    fn short_planes_refuse() {
        let layout = Biplanar::from_planes(fourcc(b"420v"), (4, 4, 1), (2, 2, 2)).unwrap();
        let mut out = [0u8; 64];
        assert!(!to_rgba8(
            layout,
            Sampling::Converted,
            &[0; 15],
            4,
            &[0; 8],
            4,
            &mut out
        ));
    }
}
