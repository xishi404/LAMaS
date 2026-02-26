from lamas.ext.lamas.scripts.evaluator import Evaluator

class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_graph_lamas(self, optimizer, directory, data, initial=False, params: dict = None):
        evaluator = Evaluator(eval_path=directory, batch_size = optimizer.batch_size)

        score, benchmark = await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            params,
            directory,
            is_test=False,
        )

        # Store benchmark instance so we can access training metrics later
        optimizer.benchmark = benchmark

        cur_round = optimizer.round
        new_data = optimizer.data_utils.create_result_data(cur_round, score)
        data.append(new_data)

        result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/train")
        optimizer.data_utils.save_results(result_path, data)

        return score
    
    async def evaluate_graph_test_lamas(self, optimizer, directory, is_test=True, params: dict = None):
        evaluator = Evaluator(eval_path=directory, batch_size = optimizer.batch_size)

        score, benchmark = await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            params,
            directory,
            is_test=is_test,
        )

        # Store benchmark instance so caller can access test metrics (tokens, cost, etc.)
        optimizer.test_benchmark = benchmark

        return score
    