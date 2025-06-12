#!/bin/sh
error() {
    >&2 printf "hw2.sh -i INPUT -o OUTPUT [-c csv|tsv] [-j]\n\Available Options:\n\n-i: Input file to be decoded\n-o: Output directory\n-c csv|tsv: Output files.[ct]sv\n-j: Output info.json"
    exit 1
}



while getopts ":i:o:c:j" opt; do
    case "$opt" in
        i)
            input="$OPTARG"
        ;;
        o)
            output="$OPTARG"
       	;;
        c)
            format="$OPTARG"
        ;;
        j)
            json=1
        ;;
        *)
            error
    esac
done

mkdir -p "$output"


if [ "$json" ] ; then
	name=$(yq '.name' "${input}")
	author=$(yq '.author' "${input}")
	date=$(yq '.date' "${input}")
	date=$(date -r "$date" -Iseconds)
	jq -r -n "{\"name\": $name,\"author\": $author,\"date\": \"$date\"}"  > "$output/info.json"
fi

error_files=0

decode(){			# input data & save_filepath
	#echo "$1"
	filepath=$(echo "$1" | yq '.name' | sed 's/"//g')
	#echo "$filepath"
	filedir=$(dirname "$filepath")
	filedir="$2/$filedir"
	filepath="$2/$filepath"
	#echo "save at $filepath"
	mkdir -p "$filedir"
	data=$(echo "$1" | yq '.data' | sed 's/"//g' | base64 -d)
	echo "$data" > "$filepath"
	sha1=$(echo "$1" | yq ".hash[\"sha-1\"]" | sed 's/"//g')
	md5=$(echo "$1" | yq ".hash[\"md5\"]" | sed 's/"//g')

	if [ "$md5" != "$(md5sum -q "$filepath")" ] || [ "$sha1" != "$(sha1sum -q "$filepath")" ]; then
        	error_files=$((error_files+1))
		echo "corrupted file"
      	fi
}

decode_all(){		# input $1 to_be_decode_filepath $2 store_dir
	file_numbers=$(yq '.files[]' "$1" | wc -l | xargs -I {} echo "{}/9-1" | bc)
	if [ "$format" = "csv" ]; then
		printf "filename,size,md5,sha1\n" > "$2/files.$format"
	elif [ "$format" = "tsv" ]; then
		printf "filename\tsize\tmd5\tsha1\n" > "$2/files.$format"
	fi
	for i in $(seq 0 "$file_numbers"); do

		full_file=$(yq ".files[$i]" "$1")			# the whole file
		file=$(yq ".files[$i][\"name\"]" "$1" | sed 's/"//g')	# file relative path to decoding
		full_file_path="$2/$file"				# file relative path to whole
		length=$(yq ".files[$i][\"data\"]" "$1" | sed 's/"//g' | base64 -d | wc -c | tr -d ' ')
		#length=$(echo "$length")

		md5=$(yq ".files[$i][\"hash\"][\"md5\"]" "$1" | sed 's/"//g')
		sha=$(yq ".files[$i][\"hash\"][\"sha-1\"]" "$1" | sed 's/"//g')
		decode "$full_file" "$2"

		if [ "$format" = "csv" ]; then
			printf '%s,%s,%s,%s\n' "$file" "$length" "$md5" "$sha" >> "$2/files.$format"
		elif [ "$format" = "tsv" ]; then
			printf "%s\t%s\t%s\t%s\n" "$file" "$length" "$md5" "$sha" >> "$2/files.$format"
		fi
		type=$(yq ".files[$i][\"type\"]" "$1" | sed 's/"//g')
		file=$(dirname "$full_file_path")
		if [ -z "$json" ] && [ -z "$format" ] && [ "$type" = "hw2" ]; then
			# echo "$file"
			decode_all "$full_file_path" "$file"
		fi
	done
}

decode_all "$input" "$output"

exit "$error_files"
