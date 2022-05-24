#! /bin/bash

for file in $1/*.chk
do
	filename=${file%".chk"}
	epsilon=${filename%_*}
	epsilon=${epsilon#*/}
	level1refinement=${filename#*_}
	echo $filename
	python table_generator.py $file $epsilon 0.5 2 $level1refinement > $filename.table
done
