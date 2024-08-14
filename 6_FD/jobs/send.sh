for file in *.batch; do
    sbatch "$file"
done