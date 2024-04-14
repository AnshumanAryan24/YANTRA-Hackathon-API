# YANTRA_Hackathon-API
Team Zephyr Algorithm Code
This repository has the tools and libraries responsible from the proper functioning of our project 'Vach Sahay'. The code in the repository AI/ML part of the project. 

## API v2:
1. This edition is to be used.
2. On-the-go detection.
3. Control parameter Z_ALPHA can be used to control accuracy.
4. Use the process() function only - functions begining from '__' are supposed to be treated as private functions

## API:
1. This is the version 1 of the API used. We have moved on to the second version of the same API.
2. API v1 has a few shortcomings which led us to move ahead with a better approach for optimisation and better features. 

## Format of JSON item:
Each element of JSON is of the format:
_(segment_start, segment_end): repetition_score_

Segment Start and End are kept constant, while the score varies and can be used to compare which parts have repetition.

### *NOTE:*
1. segment_start, segment_end are indices of segment in the numpy.array type object of wav file (when assumed, or converted, to be coming from a one channel microphone).
2. These are NOT TIME STAMPS, but INDEX STAMPS.
3. *Reason:* We will be processing .wav file as a numpy.array type, so it is convinient to have stamps in the format of indices, otherwise we have to convert every time.
