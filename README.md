# skeleton-HMM
An implementation of the paper "[Skeleton-based abnormal gait detection](http://www.mdpi.com/1424-8220/16/11/1792)" (Sensors, MDPI 2016)

## Requirements
* Python
* Numpy
* Scipy
* Scikit-learn
* hmmlearn
* Matplotlib

## Notice
* The code was implemented to directly work on [DIRO gait dataset](http://www-labs.iro.umontreal.ca/~labimage/GaitDataset/)
* Please download the [skeleton data](http://www.iro.umontreal.ca/~labimage/GaitDataset/skeletons.zip) and put the npz file into the folder **dataset**

## Usage
```python main.py -l 0 -w 5 -s 24 -o 43 -f 0```
* -l: use leave-one-out cross-validation (boolean)
* -w: width of smoothing window (int)
* -s: number of HMM's states (int)
* -o: number of HMM's observations (int)
* -f: write results to file (boolean)

## Example of output
```
test subject(s): [1 3 6 7]
Load normal gaits of 5 subjects for training...
processing normal skel. of subject 0
processing normal skel. of subject 2
processing normal skel. of subject 4
processing normal skel. of subject 5
processing normal skel. of subject 8
Load test data...
processing skel. of subject 1
processing skel. of subject 3
processing skel. of subject 6
processing skel. of subject 7
window width = 5, states = 24, observations = 43
kmeans dimension: 7
TEST RESULTS
Full sequence:   AUC = 0.898 --- EER = 0.250
Cycle:           AUC = 0.792 --- EER = 0.277
```
