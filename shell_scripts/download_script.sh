#!/bin/bash
# The base URL for the API request
base_url="https://zenodo.org/api/records/4010759"

# Fetch the JSON metadata from Zenodo
response=$(curl -s "$base_url")

# Parse JSON to extract file URLs and names using jq
echo "$response" | jq -r '.files[] | .links.self + " " + .key' | while read -r file_url file_name
do
  echo "Downloading $file_name from $file_url..."
  curl -o "$file_name" "$file_url"
  echo "$file_name downloaded."
done
echo "All files downloaded."

# Loop through all zip files in the current directory and unzip them
for file in *.zip; do
  unzip "${file%.zip}"
done
echo "All files unzipped."
