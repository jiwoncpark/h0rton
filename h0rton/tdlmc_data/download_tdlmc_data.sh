DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Rung 0
wget https://tdlmc.github.io/data/rung0.tar.gz -P $DIR
wget https://tdlmc.github.io/data/rung0_open_box.tar.gz -P $DIR
tar -xzvf $DIR/rung0.tar.gz -C $DIR
tar -xzvf $DIR/rung0_open_box.tar.gz -C $DIR
# Rung 1
wget https://tdlmc.github.io/data/rung1.tar.gz -P $DIR
wget https://tdlmc.github.io/data/rung1_open_box.zip -P $DIR
wget https://tdlmc.github.io/data/oversampled_PSF.fits -P $DIR
tar -xzvf $DIR/rung1.tar.gz -C $DIR
unzip $DIR/rung1_open_box.zip -d $DIR
# Rung 2
wget https://tdlmc.github.io/data/rung2.tar.gz -P $DIR
wget https://tdlmc.github.io/data/rung2_open_box.zip -P $DIR
tar -xzvf $DIR/rung2.tar.gz -C $DIR
unzip $DIR/rung2_open_box.zip -d $DIR
# Rung 3
wget https://tdlmc.github.io/data/rung3.tar.gz -P $DIR
wget https://tdlmc.github.io/data/rung3_open_box.zip -P $DIR
tar -xzvf $DIR/rung3.tar.gz -C $DIR
unzip $DIR/rung3_open_box.zip -d $DIR