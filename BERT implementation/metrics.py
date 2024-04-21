import datasets

rouge = datasets.load_metric("rouge")

test_Score = rouge.compute(predictions=[['test'],['test2']], references=[['test'], ['test1']], rouge_types=["rouge2"])["rouge2"].mid

print(test_Score)
