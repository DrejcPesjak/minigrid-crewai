
for s in {1..10}; do python3 -m scripts.train --algo ppo --env BabyAI-GoToRedBallGrey-v0 --model GoToRedBall-v$s --save-interval 10 --frames 250000 --seed $s | grep -v "Sampling"; done

for f in storage/GoToRedBall-v*/log.txt; do gawk '/^\[VAL512]/{ if (match($0,/SR=([0-9.]+)/,a) && a[1]>=0.99){ match($0,/total_eps=([0-9]+)/,b); match($0,/total_frames=([0-9]+)/,c); printf "%-40s SR=%s  total_eps=%s  total_frames=%s\n", FILENAME,a[1],b[1],c[1]; exit } }' "$f" || echo "$f: NO CROSSING"; done | sort


python3 -m scripts.train --algo ppo --env BabyAI-GoToRedBallGrey-v0 --model GoToRedB-vx8 --save-interval 10 --frames 200000 | grep -v "Sampling"

python3 -m scripts.evaluate --env BabyAI-GoToRedBallGrey-v0 --model GoToRedB-vx8 --episodes 512 | grep -v "Sampling"

for f in storage/GoToRedBall-v*/log.txt; do tail -n 2 $f | head -n 1; done

