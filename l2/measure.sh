echo "" >./results.txt
export OMP_NESTED=TRUE
for steps in 100 1000 10000 100000; do
    for how in REDUCTION CRITICAL SEQUENTIAL; do
        for threads in 64; do
            g++ -fopenmp pi.cpp -o pi -D"$how" -DSTEPS="$steps" -DTHREADS="$threads"
            echo "$threads" "$steps" "$how"
            echo -n "$threads $steps $how " >>./results.txt
            /usr/bin/time -f "%U" ./pi 2>>./results.txt >/dev/null
        done
    done
done
