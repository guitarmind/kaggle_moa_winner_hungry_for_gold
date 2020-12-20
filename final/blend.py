#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Blending best single models')
parser.add_argument('input', metavar='INPUT',
                    help='Input folder', default=".")
parser.add_argument('predictions', metavar='PREDS',
                    help='Prediction file folder', default=".")
parser.add_argument('output', metavar='OUTPUT',
                    help='Output folder', default=".")

args = parser.parse_args()
input_folder = args.input
prediction_folder = args.predictions
output_folder = args.output

print("Generating winning solution submission file ......")

sub1 = pd.read_csv(f'{prediction_folder}/submission_3stage_nn_0.01822.csv')
sub2 = pd.read_csv(f'{prediction_folder}/submission_2heads_resnet_0.01836.csv')
sub3 = pd.read_csv(f'{prediction_folder}/submission_simpleNN_newcv_0.01830.csv')
sub4 = pd.read_csv(f'{prediction_folder}/submission_effnet_v7_b3.csv')
sub5 = pd.read_csv(f'{prediction_folder}/submission_resnest_v2.csv')
sub6 = pd.read_csv(f'{prediction_folder}/submission_2stage_nn_tabnet_0.01837.csv')
sub7 = pd.read_csv(f'{prediction_folder}/submission_simpleNN_newcv_0.01830.csv')

submission = pd.read_csv(f'{input_folder}/sample_submission.csv')
submission.iloc[:, 1:] = 0
submission.iloc[:, 1:] = (sub1.iloc[:,1:]*0.37 + \
    sub3.iloc[:,1:]*0.1 + sub4.iloc[:,1:]*0.18 + sub5.iloc[:,1:]*0.15)*0.9 + \
    sub6.iloc[:,1:]*0.1 + \
    sub7.iloc[:,1:]*0.09 + \
    sub2.iloc[:,1:]*0.09

submission.to_csv(f'{output_folder}/winning_submission.csv', index=False)
print("DONE!")
