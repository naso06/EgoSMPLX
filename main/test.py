import argparse
from config import cfg
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import os.path as osp
from pdb import set_trace
import time



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--result_path', type=str, default='output/test')
    parser.add_argument('--ckpt_idx', type=int, default=0)
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', type=str, default='na')
    parser.add_argument('--shapy_eval_split', type=str, default='val')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--eval_on_train', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_feature', action='store_true')
    parser.add_argument('--vis_hand_bbox', action='store_true')
    parser.add_argument('--vis_2d_pose', action='store_true')
    args = parser.parse_args()
    return args

def main():
    print('### Argument parse and create log ###')
    args = parse_args()

    config_path = osp.join('../output',args.result_path, 'code', 'config_base.py')
    ckpt_path = osp.join('../output', args.result_path, 'model_dump', f'snapshot_{int(args.ckpt_idx)}.pth.tar')
    # set_trace()

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(args.testset, args.agora_benchmark, args.shapy_eval_split, 
                           ckpt_path, args.use_cache, args.eval_on_train, args.vis)
    cfg.update_config(args.num_gpus, args.exp_name)
    cfg.vis_feature = args.vis_feature
    cfg.vis_hand_bbox = args.vis_hand_bbox
    cfg.vis_2d_pose = args.vis_2d_pose

    cudnn.benchmark = True

    from base import Tester
    tester = Tester()


    tester._make_batch_generator()
    tester._make_model()
    start_time = time.time()
    frame_count = 0

    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        iter_start = time.time()

        # forward
        with torch.no_grad():
            model_out = tester.model(inputs, targets, meta_info, 'test')

        iter_end = time.time()
        iter_time = iter_end - iter_start
        fps = 1.0 / iter_time if iter_time > 0 else 0
        print(f"[Iter {itr}] time: {iter_time:.4f}s  FPS: {fps:.2f}")
        frame_count += 1


        # save output
        batch_size = model_out['img'].shape[0]
        out = {}
        for k, v in model_out.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu().numpy()
            elif isinstance(v, list):
                out[k] = v
            else:
                raise ValueError('Undefined type in out. Key: {}; Type: {}.'.format(k, type(v)))
        # out = {k: v.cpu().numpy() for k, v in out.items()}
        # for k, v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k, v in cur_eval_result.items():
            if k in eval_result:
                eval_result[k] += v
            else:
                eval_result[k] = v
        cur_sample_idx += len(out)
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    print(f"Average FPS: {avg_fps:.2f}")


    tester._print_eval_result(eval_result)

if __name__ == "__main__":
    main()