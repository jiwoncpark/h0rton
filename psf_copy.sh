for d in data/rung1/code*/f160w-seed*; do cp -- "$d/drizzled_image/psf.fits" "psf_maps/psf_${d: -3}.fits"; done
