#!/bin/bash

base_command="huggingface-cli download "

# Function to display help information
print_help() {
  echo "Usage: $0 <repo_id> [-d <local_dir>] [optional_args...]"
  echo ""
  echo "Arguments:"
  echo "  repo_id                 The ID of the repository to download."
  echo "  optional_args           Additional optional arguments to specify specific files."
  echo ""
  echo "Options:"
  echo "  -h, --help              Show this help message and exit."
  echo "  -d <local_dir>          Specify a local directory for the download."
  echo ""
  echo "Example:"
  echo "  $0 some-repo-id -d ./local_data <file_name>"
  exit 1  # Exit with an error status
}

# Check if the first argument is a flag instead of a positional argument (repo_id)
if [[ -z $1 ]] || [[ $1 == -* ]]; then
  print_help
fi

repo_id=$1
shift # Shift the positional parameters to the left, removing the repo_id from the process

# Initialize an array to hold optional arguments
declare -a optional_args=()

# Parse the rest of the arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -d)
      if [[ -n $2 ]] && [[ $2 != -* ]]; then
        local_dir=$2
        base_command+=" --repo-type dataset --local-dir $local_dir --local-dir-use-symlinks False"
        shift 2
      else
        echo "Error: '-d' requires a non-empty option argument."
        exit 1
      fi
      ;;
    *) # Collect all other parameters as optional arguments
      optional_args+=("$1")
      shift
      ;;
  esac
done

# Append repo_id to the base command
base_command+=" $repo_id"

# Append all optional arguments to the end of the base command
for arg in "${optional_args[@]}"; do
  base_command+=" $arg"
done

echo "Running command: $base_command"

# Command to run huggingface-cli download
export HF_HUB_ENABLE_HF_TRANSFER=0
# cmd='huggingface-cli download --resume-download --repo-type dataset --local-dir data/openorca/src --local-dir-use-symlinks False Open-Orca/OpenOrca'

while true; do
  # Run the command in the background and get its PID
  $base_command 2>&1 & cmd_pid=$!

  # Monitor the background process for the keyword
  { grep --line-buffered -q "Traceback" <(tail -f --pid=$cmd_pid /proc/$cmd_pid/fd/1 2>/dev/null); kill $cmd_pid; } & monitor_pid=$!

  # Wait for the command to exit
  wait $cmd_pid
  exit_status=$?

  # Kill the monitor process in case the command exits before the keyword is found
  kill $monitor_pid 2>/dev/null

  # Check exit status of the command
  if [ $exit_status -eq 0 ]; then
    # Command succeeded, exit the loop
    break
  else
    # Command failed, restart it
    echo -e "\e[1;33mThe command failed, restarting after 1 second...\e[0m"
    sleep 1
  fi
done