from Matrix_Generator.standardize_and_concatenate import main_workflow as standardized_concatenate

def main():
    # Get the standardized concatenated dataframe containing all of the data
    data_df = standardized_concatenate()

    # Generate pairwise position-weighted matrices
    


# If the script is executed directly, invoke the main workflow
if __name__ == "__main__":
    main()