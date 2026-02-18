import subprocess

if __name__ == "__main__":
    benchmarks = [
        # "aieducation/what-course-are-you-going-to-take/",
        # "ampiiere/animal-crossing-villager-popularity-analysis/",
        # "dataranch/supermarket-sales-prediction-xgboost-fastai/",
        # "erikbruin/nlp-on-student-writing-eda/",
        # "gksriharsha/eda-speedtests/",
        # "ibtesama/getting-started-with-a-movie-recommendation-system/",
        # "josecode1/billionaires-statistics-2023/",
        # "joshuaswords/netflix-data-visualization/",
        # "kkhandekar/environmental-vs-ai-startups-india-eda/",
        # "lextoumbourou/feedback3-eda-hf-custom-trainer-sift/",
        # "madhurpant/beautiful-kaggle-2022-analysis/",
        # "mpwolke/just-you-wait-rishi-sunak/",
        # "nickwan/creating-player-stats-using-tracking-data/",
        # "paultimothymooney/kaggle-survey-2022-all-results/",
        # "pmarcelino/comprehensive-data-exploration-with-python/",
        # "roopacalistus/retail-supermarket-store-analysis/",
        # "saisandeepjallepalli/adidas-retail-eda-data-visualization/",
        # "sandhyakrishnan02/indian-startup-growth-analysis/",
        # "sanket7994/imdb-dataset-eda-project/",
        # "spscientist/student-performance-in-exams/",
    ]
    small_notebook_template = "/home/colinc/code/dias-benchmarks/notebooks/{benchmark_name}src/small_bench.ipynb"
    full_notebook_template = (
        "/home/colinc/code/dias-benchmarks/notebooks/{benchmark_name}src/bench.ipynb"
    )

    for benchmark in benchmarks:
        small_notebook = small_notebook_template.format(benchmark_name=benchmark)
        full_notebook = full_notebook_template.format(benchmark_name=benchmark)
        # subprocess.run(["python", "utils/main_cpu.py", "--small_notebook_path", small_notebook, "--full_notebook_path", full_notebook])
        try:
            subprocess.run(
                [
                    "python",
                    "utils/main_cpu.py",
                    "--small_notebook_path",
                    small_notebook,
                    "--full_notebook_path",
                    full_notebook,
                ]
            )
        except subprocess.TimeoutExpired:
            print("Process timed out.")
        print(f"{small_notebook} rewritten done")

    new_notebook_path = [
        # "/home/colinc/code/dias-benchmarks/new_notebooks/imdb",
        # "/home/colinc/code/dias-benchmarks/new_notebooks/nyc-airbnb",
        # "/home/colinc/code/dias-benchmarks/new_notebooks/nyc-flight",
        # "/home/colinc/code/dias-benchmarks/new_notebooks/nyc-taxi",
        # "/home/colinc/code/dias-benchmarks/new_notebooks/us-birth",
    ]
    for p in new_notebook_path:
        small_notebook = f"{p}/small_bench.ipynb"
        full_notebook = f"{p}/bench.ipynb"
        try:
            subprocess.run(
                [
                    "python",
                    "utils/main_cpu.py",
                    "--small_notebook_path",
                    small_notebook,
                    "--full_notebook_path",
                    full_notebook,
                ]
            )
        except subprocess.TimeoutExpired:
            print("Process timed out.")

        print(f"{small_notebook} rewritten done")

    tpch_notebook_path = [
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q01/q01_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q02/q02_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q03/q03_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q04/q04_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q05/q05_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q06/q06_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q07/q07_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q08/q08_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q09/q09_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q10/q10_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q11/q11_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q12/q12_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q13/q13_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q14/q14_rewrite.ipynb",
        # "/home/colinc/code/dias-benchmarks/tpch/notebooks/q15/q15_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q16/q16_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q17/q17_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q18/q18_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q19/q19_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q20/q20_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q21/q21_rewrite.ipynb",
        "/home/colinc/code/dias-benchmarks/tpch/notebooks/q22/q22_rewrite.ipynb",
    ]

    for p in tpch_notebook_path:
        small_notebook = p
        full_notebook = p
        try:
            subprocess.run(
                [
                    "python",
                    "utils/main_cpu.py",
                    "--small_notebook_path",
                    small_notebook,
                    "--full_notebook_path",
                    full_notebook,
                ]
            )
        except subprocess.TimeoutExpired:
            print("Process timed out.")

        print(f"{small_notebook} rewritten done")
