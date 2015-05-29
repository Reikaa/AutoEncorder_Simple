#!/usr/bin/env bash
find average_err/ average_time/ distributions/ annotated/ neurons/ reconstruction/ individual_time/ individual_err/ data_set_err/ data_set_time/ -name '*.svg' | xargs readlink -f | xargs -I {} echo ./node_modules/phantomjs/bin/phantomjs rasterise.js file:///{} {}.png
