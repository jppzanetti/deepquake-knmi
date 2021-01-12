## Setup

This project uses Python 3, and Tensorflow 2.

The first step to set it up is to install the required Python
libraries. These commands and the all the ones that follow are to be
run from the `deepquake/` directory:

```
pip install --upgrade pip
pip install -r requirements.txt
```

Some configuration variables also need to be filled. Rename
`deepquake/local_config.py.sample` to
`deepquake/local_config.py`, and edit it, especially `ROOT_PATH`,
with the full path of the `deepquake/` directory, and the MongoDB
credentials (for storing prediction results).

## Getting input metadata

The input event and station information can be retrieved from FDSN
webservices. For example, how to get input data from 2014 to 2018:

```
wget "http://rdsa.knmi.nl/fdsnws/event/1/query?starttime=2014-01-01&endtime=2019-01-01" -O inputdata/event_catalog.xml
wget "http://rdsa.knmi.nl/fdsnws/station/1/query?level=response&starttime=2014-01-01&endtime=2019-01-01" -O inputdata/NL_stations_2014-2018.xml
```

## Get samples

To fetch sample data use the `deepquake/preprocess/get_streams.py`
script. It gets streams either from FDSN webservices or local files,
processes them, and saves them in `.tfrecords` files. Before running
it, it is necessary to edit the `SDS_ROOT` variable in the
`deepquake/preprocess/stream.py` file with the path (to be) used
for locally storing MSEED files in a SDS-like directory structure.

To split the samples into train/test/evaluation sets, it is possible
to use the `scripts/split_sets.py`. Note that this might not the ideal
way to do it, because this script can put in separate sets samples
from different stations but corresponding to the same event.

Example:

```
python deepquake/preprocess/get_streams.py --inventory inputdata/NL_stations_2014-2018.xml --events inputdata/event_catalog.xml --output streams/ --event_fraction <0.01~1.0> --use-fdsn
python scripts/split_sets.py -i streams/filter/0 -o dataset/noise/
python scripts/split_sets.py -i streams/filter/1 -o dataset/events/
python scripts/split_sets.py -i streams/filter/2 -o dataset/events/
```

## Train model

Example:

```
python deepquake/train.py --feature detection --num_epochs <N>
```

## Evaluate model

Example:

```
python deepquake/evaluate.py --feature detection <saved_weights_file.h5>
```

## Predict

TODO: arguments

Example:

```
python deepquake/predict.py
```
