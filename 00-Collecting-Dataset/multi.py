import multiprocessing
import subprocess

def run_scraper(instance_id, start_id, end_id):
    subprocess.run([
        "python", "DataSetScrapper.py",
        str(instance_id),
        str(start_id),
        str(end_id)
    ])

if __name__ == "__main__":
    ranges = [
        (1, 1000000, 1200000),
        (2, 1200000, 1400000),
        (3, 1400000, 1600000),
        (4, 1600000, 1800000),
        (5, 1800000, 2000000),
        (6, 2000000, 2200000),
        (7, 2200000, 2400000),
        (8, 2400000, 2600000),
        (9, 2600000, 2800000),
        (10, 2800000, 3000000),
    ]
    pool = multiprocessing.Pool(processes=5)
    pool.starmap(run_scraper, ranges)