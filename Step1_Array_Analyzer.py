#This script computes log2fc and does significance testing for the SPOT blot densitometry data.
#SPOT peptide synthesis is a method in which peptides are synthesized directly onto cellulose membranes...
#... and probed with recombinant bait proteins detected by a fluorescent reporter. Fluorescence is...
#... logarithmically related to binding affinity. 

#Import required functions and packages

import numpy as np
import pandas as pd
import math
import os
from PACM_General_Functions import FilenameSubdir, Log2fc, BaitPermuter, ListInputter

print("----------------")
print("Script 1:")
print("This script is performing comparisons on SPOT densitometry data.")
print("----------------")

#Read the SPOT densitometry data. This script assumes there are 2 replicates.

dens_source_filename = input("Please enter the filename for your CSV containing SPOT densitometry data:  ")

dens_df = pd.read_csv(dens_source_filename)

#Write the bait list here; it will be permuted for various calculations

more_baits = True
list_of_baits = ListInputter("Please input the baits you want to analyze one at a time and hit enter when done.")

#Standardization

standardize = input("Would you like to standardize this data? (Y/N)  ")
if standardize == "Y": 
	hit_for_standardization = input("Which hit do you want to use for standardization?  ")
	stand_indices = dens_df.index[dens_df["Name"]==hit_for_standardization].tolist() #Gets the list of indices matching user input
	stand_index = stand_indices[0] #Selects the first instance to use

	standardization_loop_baits = list_of_baits.copy()
	standardization_loop_baits.append("Control") #Ensures control values will also be standardized

	#Compute average value of the user-inputted standard across baits
	counter_standard = 0
	sum_standard = 0
	for bait in list_of_baits: 
		sum_standard = sum_standard + dens_df.at[stand_index, bait + "_1"] + dens_df.at[stand_index, bait + "_2"]
		counter_standard = counter_standard + 2
	average_standard = sum_standard / counter_standard

	for i in np.arange(len(dens_df)): 
		for bait in standardization_loop_baits: 
			old_value_1 = dens_df.at[i, bait + "_1"]
			old_value_2 = dens_df.at[i, bait + "_2"]

			standard_1 = dens_df.at[stand_index, bait + "_1"]
			standard_2 = dens_df.at[stand_index, bait + "_2"]

			if standard_1 > 0: 
				new_value_1 = (old_value_1 / standard_1) * average_standard
				dens_df.at[i, bait + "_1"] = new_value_1

			if standard_2 > 0: 
				new_value_2 = (old_value_2 / standard_2) * average_standard
				dens_df.at[i, bait + "_2"] = new_value_2

	print("Standardization using", hit_for_standardization, "is complete!")
	print("----------------")

#-------------------------------------------------------------------------------------------------

#Calculation of fold change (log2fc) between each pair of hits

log2fc_columns = BaitPermuter(list_of_baits, "log2fc")

#Construct empty DataFrame to contain log2fc values for the previous columns

log2fc_df = pd.DataFrame(index = np.arange(len(dens_df)), columns = log2fc_columns)
log2fc_df = log2fc_df.fillna(0)

#Conditionally compute log2fc while aware of controls

def ConditionalLog2fc(first_bait, second_bait, source_dataframe, dest_dataframe, control_multiplier_val):
	if first_bait != second_bait and source_dataframe.loc[i, first_bait + "_Total_Pass"] == "NaN" and source_dataframe.loc[i, second_bait + "_Total_Pass"] == "NaN": 
		# Tests if at least one bait passes manual qualitative analysis; if neither do, assigns log2fc as NaN
		dest_dataframe.loc[i, first_bait + "_" + second_bait + "_log2fc"] = "NaN"
	elif first_bait != second_bait: 
		sum_controls = source_dataframe.loc[i, "Control_1"] + source_dataframe.loc[i, "Control_2"]
		sum_multiplier_controls = control_multiplier_val * sum_controls

		sum_bait1 = source_dataframe.loc[i, first_bait + "_"+str(1)] + source_dataframe.loc[i, first_bait + "_"+str(2)]
		sum_bait2 = source_dataframe.loc[i, second_bait + "_"+str(1)] + source_dataframe.loc[i, second_bait + "_"+str(2)]

		if sum_bait1 > sum_multiplier_controls or sum_bait2 > sum_multiplier_controls: #Requires at least one of the sums to exceed a multiple of the control value based on control_multiplier_val
			mean_bait1 = sum_bait1 / 2
			mean_bait2 = sum_bait2 / 2				
		else: 
			mean_bait1 = 0
			mean_bait2 = 0

		log2fc_bait1_bait2 = Log2fc(mean_bait1, mean_bait2)

		dest_dataframe.loc[i, first_bait + "_" + second_bait + "_log2fc"] = log2fc_bait1_bait2

control_multiplier = NumInput("Log2fc values are returned when the two compared values are above a defined multiple of the control to prevent erroneous results. A multiplier between 2 and 5 is advised. Enter the control multiplier:")
for i in np.arange(len(dens_df)): 
	for bait1 in list_of_baits: 
		for bait2 in list_of_baits: 
			ConditionalLog2fc(bait1, bait2, dens_df, log2fc_df, control_multiplier)

dens_log2fc_df = pd.concat([dens_df, log2fc_df], axis = 1)

#-------------------------------------------------------------------------------------------------

#Mark whether peptides bind at least one bait based on either: 
#... (1) Manual calls when reading the SPOT blots by eye, or 
#... (2) Automatic calls by comparing to control values
#CAUTION: SPOT blots tend to vary in background across the blot, so automatic calls may miss low-level binders.

one_passes_df = pd.DataFrame(index = np.arange(len(dens_log2fc_df)), columns = ["One_Passes"])

def OnePassesManual(bait_list, input_dataframe, output_dataframe, output_col_name):
	final_df = output_dataframe.copy()
	for i in np.arange(len(input_dataframe)): 
		one_passes = "No"
		for bait in bait_list: 
			if input_dataframe.at[i, bait + "_Total_Pass"] == "Pass":
				one_passes = "Yes"
			elif input_dataframe.at[i, bait + "_Total_Pass"] == "Borderline" and one_passes == "No":
				one_passes = "Maybe"
		final_df.at[i, output_col_name] = one_passes
	return final_df

def OnePassesAuto(bait_list, input_dataframe, output_dataframe, output_col_name, multiplier):
	final_df = output_dataframe.copy()
	for i in np.arange(len(input_dataframe)): 
		one_passes = "No"
		for bait in bait_list: 
			sum_bait = input_dataframe.at[i, bait + "_1"] + input_dataframe.at[i, bait + "_2"]
			sum_controls = input_dataframe.at[i, "Control_1"] + input_dataframe.at[i, "Control_2"]
			sum_multiplier_controls = multiplier * sum_controls
			if sum_bait > sum_multiplier_controls:
				one_passes = "Yes"
			elif sum_bait > sum_controls:
				one_passes = "Maybe"
		final_df.at[i, output_col_name] = one_passes
	return final_df

one_passes_auto_df = OnePassesAuto(list_of_baits, dens_log2fc_df, one_passes_df, "One_Passes_Auto", control_multiplier)

call_method = input("Would you like to use manual calls for whether spots pass as positive? (Y/N)  ")
if call_method == "Y": 
	one_passes_manual_df = OnePassesManual(list_of_baits, dens_log2fc_df, one_passes_df, "One_Passes_Manual")	
	dens_log2fc_logical_df = pd.concat([dens_log2fc_df, one_passes_manual_df, one_passes_auto_df], axis = 1)
else: 
	dens_log2fc_logical_df = pd.concat([dens_log2fc_df, one_passes_auto_df], axis = 1)

#-------------------------------------------------------------------------------------------------

#Calculation of max bait

max_df = pd.DataFrame(index = np.arange(len(dens_log2fc_logical_df)), columns = ["Max_Bait_Mean", "Max_Bait_div_Control"])

for i, row in dens_log2fc_logical_df.iterrows(): 
	mean_controls = (dens_log2fc_logical_df.loc[i, "Control_1"] + dens_log2fc_logical_df.loc[i, "Control_2"]) / 2

	max_bait_uncontrolled = 0
	max_bait = 0

	for bait in list_of_baits: 
		current_bait_uncontrolled = (dens_log2fc_logical_df.loc[i, bait + "_1"] + dens_log2fc_logical_df.loc[i, bait + "_2"]) / 2
		if current_bait_uncontrolled > max_bait_uncontrolled: 
			max_bait_uncontrolled = current_bait_uncontrolled
			if mean_controls > 0: 
				max_bait_fold = str(max_bait_uncontrolled / mean_controls)
			elif mean_controls == 0 and max_bait_uncontrolled > 0: 
				max_bait_fold = "Inf"

	max_df.at[i, "Max_Bait_Mean"] = max_bait_uncontrolled
	max_df.at[i, "Max_Bait_div_Control"] = max_bait_fold

dens_analyzed_df = pd.concat([dens_log2fc_logical_df, max_df], axis = 1)

#-------------------------------------------------------------------------------------------------

#Significance testing

significance_df = pd.DataFrame(index = np.arange(len(dens_analyzed_df)), columns = ["Significant"])

if call_method = "Y":
	for i, row in dens_analyzed_df.iterrows(): 
		significant = "NaN"
		list_passes = [dens_analyzed_df.at[i, "One_Passes_Manual"], dens_analyzed_df.at[i, "One_Passes_Auto"]]
		if list_passes == ["Yes", "Yes"]: 
			significant = "Yes"
		elif list_passes == ["Yes", "Maybe"] or list_passes == ["Maybe", "Yes"] or list_passes == ["Maybe", "Maybe"]:
			significant = "Maybe"
		else: 
			significant = "No"
		significance_df.at[i, "Significant"] = significant

dens_final_df = pd.concat([dens_analyzed_df, significance_df], axis = 1)

#Save to output folder

dens_final_df.to_csv(FilenameSubdir("Output", "Densitometry_Analyzed_Results.csv"))

print("Saved! Filename: Output/Densitometry_Analyzed_Results.csv")
print("-------------------")