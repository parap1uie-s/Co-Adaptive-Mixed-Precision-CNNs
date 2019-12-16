import distiller
import operator
import os

class missingdict(dict):
    """This is a little trick to prevent KeyError"""
    def __missing__(self, key):
        return None  # note, does *not* set self[key] - we don't want defaultdict's behavior

def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.
    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels
    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test
    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    distiller.utils.assign_layer_fq_names(model)

    genCollectors = lambda: missingdict({
        "sparsity":      SummaryActivationStatsCollector(model, "sparsity",
                                                         lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}

def update_training_scores_history(perf_scores_history, model, top1, top5, epoch, num_best_scores, msglogger):
    """ Update the list of top training scores achieved so far, and log the best scores so far"""

    model_sparsity, _, params_nnz_cnt = distiller.model_params_stats(model)
    perf_scores_history.append(distiller.MutableNamedTuple({'params_nnz_cnt': -params_nnz_cnt,
                                                            'sparsity': model_sparsity,
                                                            'top1': top1, 'top5': top5, 'epoch': epoch}))
    # Keep perf_scores_history sorted from best to worst
    # Sort by sparsity as main sort key, then sort by top1, top5 and epoch
    perf_scores_history.sort(key=operator.attrgetter('params_nnz_cnt', 'top1', 'top5', 'epoch'), reverse=True)
    for score in perf_scores_history[:num_best_scores]:
        msglogger.info('==> Best [Top1: %.3f   Top5: %.3f   Sparsity:%.2f   Params: %d on epoch: %d]',
                       score.top1, score.top5, score.sparsity, -score.params_nnz_cnt, score.epoch)

def save_collectors_data(collectors, directory, msglogger):
    """Utility function that saves all activation statistics to Excel workbooks
    """
    for name, collector in collectors.items():
        workbook = os.path.join(directory, name)
        msglogger.info("Generating {}".format(workbook))
        collector.save(workbook)