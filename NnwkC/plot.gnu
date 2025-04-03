set terminal png

set output "Lpfnt.png"
set title "Losses for training and testing MLP"
set xlabel "Loss"
set ylabel "Epoch"
set xrange[-1:]
plot "losses.txt" using 1:2 title "Training" with lines, "losses.txt" using 1:3 title "Testing" with lines
