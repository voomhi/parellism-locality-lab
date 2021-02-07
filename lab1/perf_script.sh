#!/bin/bash

prog_file=""
out_file=""
events="instructions cycles branches branch-misses cache-misses cache-references"
while [ "$1" != "" ];
 do
    case $1 in
	-prog  | --program )  shift
			      prog_file=$1
			      ;;
	-o  | --output )
	    shift
	    out_file=$1
	    ;;
	-e   | --events )
	    shift
	    events="$1 $events"
	    ;;            
	# *)
	#     exit 1 # error
        #     ;;
    esac
    shift
done
# if [ "$prog_file" = "" ]
# then
#     exit
# fi
# if [ "$out_file" = "" ]
# then
#     exit
# fi

for event in $events
do
    perf stat -x, -e $event -o test.txt --append $PWD/basecode 
done


#Captured parameters
#cache-misses
#branches
#branch-misses
#cycles


