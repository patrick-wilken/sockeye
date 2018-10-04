# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Translation CLI.
"""
import argparse
import sys
import time
from contextlib import ExitStack
from math import ceil
from typing import Generator, Optional, List

import sockeye
import sockeye.arguments as arguments
import sockeye.constants as C
import sockeye.data_io
from sockeye import inference
from sockeye.lexicon import TopKLexicon
from sockeye.log import setup_main_logger
from sockeye.output_handler import get_output_handler, OutputHandler
from sockeye.utils import determine_context, log_basic_info, check_condition, grouper

logger = setup_main_logger(__name__, file_logging=False)


def init(config: str):
    params = arguments.ConfigArgumentParser(description='Translate CLI')
    arguments.add_translate_cli_args(params)
    args = params.parse_args(['--config', config])

    if args.output is not None:
        global logger
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))

    if args.checkpoints is not None:
        check_condition(len(args.checkpoints) == len(args.models), "must provide checkpoints for each model")

    if args.skip_topk:
        check_condition(args.beam_size == 1, "--skip-topk has no effect if beam size is larger than 1")
        check_condition(len(args.models) == 1, "--skip-topk has no effect for decoding with more than 1 model")

    log_basic_info(args)

    with ExitStack() as exit_stack:
        check_condition(len(args.device_ids) == 1, "translate only supports single device for now")
        context = determine_context(device_ids=args.device_ids,
                                    use_cpu=args.use_cpu,
                                    disable_device_locking=args.disable_device_locking,
                                    lock_dir=args.lock_dir,
                                    exit_stack=exit_stack)[0]
        logger.info("Translate Device: %s", context)

        if args.override_dtype == C.DTYPE_FP16:
            logger.warning('Experimental feature \'--override-dtype float16\' has been used. '
                           'This feature may be removed or change its behaviour in future. '
                           'DO NOT USE IT IN PRODUCTION!')

        models, source_vocabs, target_vocab = inference.load_models(
            context=context,
            max_input_len=args.max_input_len,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            model_folders=args.models,
            checkpoints=args.checkpoints,
            softmax_temperature=args.softmax_temperature,
            max_output_length_num_stds=args.max_output_length_num_stds,
            decoder_return_logit_inputs=args.restrict_lexicon is not None,
            cache_output_layer_w_b=args.restrict_lexicon is not None,
            override_dtype=args.override_dtype)
        restrict_lexicon = None  # type: Optional[TopKLexicon]
        if args.restrict_lexicon:
            restrict_lexicon = TopKLexicon(source_vocabs[0], target_vocab)
            restrict_lexicon.load(args.restrict_lexicon, k=args.restrict_lexicon_topk)
        store_beam = args.output_type == C.OUTPUT_HANDLER_BEAM_STORE
        translator = inference.Translator(context=context,
                                          ensemble_mode=args.ensemble_mode,
                                          bucket_source_width=args.bucket_width,
                                          length_penalty=inference.LengthPenalty(args.length_penalty_alpha,
                                                                                 args.length_penalty_beta),
                                          beam_prune=args.beam_prune,
                                          beam_search_stop=args.beam_search_stop,
                                          models=models,
                                          source_vocabs=source_vocabs,
                                          target_vocab=target_vocab,
                                          restrict_lexicon=restrict_lexicon,
                                          avoid_list=args.avoid_list,
                                          store_beam=store_beam,
                                          strip_unknown_words=args.strip_unknown_words,
                                          skip_topk=args.skip_topk)

    return translator


def make_chunk_from_plain(source_data: List[str]):
    chunk = []
    for sentence_id, line in enumerate(source_data, 1):
        chunk.append(inference.make_input_from_plain_string(sentence_id, line))
    return chunk


def translate(source_data: List[str],
              translator: inference.Translator,
              output_handler: OutputHandler,) -> float:
    """
    Translates each line from source_data, calling output handler after translating a batch.

    :param output_handler: A handler that will be called once with the output of each translation.
    :param trans_inputs: A enumerable list of translator inputs.
    :param translator: The translator that will be used for each line of input.
    :return: Total time taken.
    """
    trans_inputs = make_chunk_from_plain(source_data)
    tic = time.time()
    trans_outputs = translator.translate(trans_inputs)
    total_time = time.time() - tic
    batch_time = total_time / len(trans_inputs)
    for trans_input, trans_output in zip(trans_inputs, trans_outputs):
        output_handler.handle(trans_input, trans_output, batch_time)
    return total_time
