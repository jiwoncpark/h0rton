DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
wget https://tdlmc.github.io/data/rung0.tar.gz -P $DIR
wget https://tdlmc.github.io/data/rung0_open_box.tar.gz -P $DIR
wget https://tdlmc.github.io/data/oversampled_PSF.fits -P $DIR
wget https://tdlmc.github.io/data/rung2.tar.gz -P $DIR
wget https://tdlmc.github.io/data/rung2_open_box.zip -P $DIR
wget https://tdlmc.github.io/data/rung3.tar.gz -P $DIR
wget https://tdlmc.github.io/data/rung3_open_box.zip -P $DIR
tar -xzvf rung0.tar.gz rung0_open_box.tar.gz rung2.tar.gz rung3.tar.gz -C $DIR
unzip rung2_open_box.zip rung3_open_box.zip -d $DIR
tar -xzvf $DIR/rung0.tar.gz -C $DIR