#!/bin/bash

# Check if a directory path is provided as an argument
if [[ -z "$1" ]]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Assign the provided directory path to a variable
directory_path="$1"

# Check if the provided path is a valid directory
if [[ ! -d "$directory_path" ]]; then
    echo "Error: '$directory_path' is not a valid directory."
    exit 1
fi

# TODO: if there are multiple runs for the same job id, take the latest one into account.

# Report jobs that were cancelled due to time limit
nanloss_job_ids=()
cancelled_job_ids=()
incomplete_job_ids=()
# Iterate over all files in the specified directory
for file in "$directory_path"/*; do
    # Check if the current item is a file
    if [[ -f "$file" ]]; then
        # Extract the last line of the file
        last_line=$(tail -n 1 "$file")
        # Check if the third line from the end contains "====> Epoch: 349 Average loss"
        third_last_line=$(tail -n 3 "$file" | head -n 1)
        
        # Check if the file contains "Loss: nan"
        if grep -q "Loss: nan" "$file"; then
            # Extract the job ID for jobs with nan loss
            job_id=$(echo "$file" | grep -oP '\d+(?=\.out$)')
            nanloss_job_ids+=("$job_id")
        # Check if the last line contains both "cancelled" and "due to time limit"
        elif [[ "$last_line" == *"CANCELLED"* && "$last_line" == *"DUE TO TIME LIMIT"* ]]; then
            # Extract the job ID from the filename (assuming it's part of the filename)
            # Example: If the filename is "slurm-12345.out", extract "12345"
            job_id=$(echo "$file" | grep -oP '\d+(?=\.out$)')
            
            last_epoch="$(tail -n 7 $file | head -n 1 | grep -oP '(?<=Train Epoch: )\d+')"
            echo "$job_id: $last_epoch"
            # Print the job ID
            # echo "Job ID $job_id was cancelled due to time limit."
            # echo -n "$job_id, "
            cancelled_job_ids+=("$job_id")
        # check if the job is incomplete due to other reasons than time limit
        elif [[ "$third_last_line" != *"====> Epoch: 349 Average loss"* ]]; then
            # incomplete_job_ids+=("$file")
            job_id=$(echo "$file" | grep -oP '\d+(?=\.out$)')
            incomplete_job_ids+=("$job_id")
        fi
    fi
done

echo "Nan loss jobs (IDs):"
printf '%s\n' "${nanloss_job_ids[@]}" | sort -n | paste -sd ',' -
nanloss_count=${#nanloss_job_ids[@]}
echo "Total jobs due with nan loss: $nanloss_count"
echo ""

echo "Cancelled jobs (IDs):"
printf '%s\n' "${cancelled_job_ids[@]}" | sort -n | paste -sd ',' -
cancelled_count=${#cancelled_job_ids[@]}
echo "Total cancelled jobs due to time limit: $cancelled_count"
echo "Instruction: you can simply copy paste the above job IDs and pass it to the 'sbatch' in the submit_array.sh script to rerun the jobs."
echo ""

# Report jobs that did not finish successfully due to other reasons
echo "Other incomplete jobs (IDs):"
printf '%s\n' "${incomplete_job_ids[@]}" | sort -n | paste -sd ',' -
incomplete_count=${#incomplete_job_ids[@]}
echo "Instruction: you can investigate the logs of these jobs to find out why they did not finish successfully."
echo "Total incomplete jobs due to unknown reason: $incomplete_count"

