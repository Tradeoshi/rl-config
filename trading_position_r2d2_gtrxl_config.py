from easydict import EasyDict
import torch
collector_env_num = 1
evaluator_env_num = 1
priority=True,
priority_IS_weight=False,
trading_position_r2d2_gtrxl_config = dict(
    exp_name='trading_position_r2d2_gtrxl_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,

        positions=[-1, 0, 1],
        # Prev number of kline will keep obs and LSTM NN will use it for choose the action of next step
        windows=None,
        trading_fees=0.0003,
        borrow_interest_rate=0,
        portfolio_initial_value=1000000,
        initial_position="random",
        start_date='2021-08-01',
        end_date='2023-09-24',
        train_range=0.7,
        test_range=0.3,
        trading_currency='BTCUSDT',
        indicators=['close_9_ema', 'close_21_ema', 'macd', 'atr_14', 'obv', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower'],
        is_train=True,
        is_render=False,
        verbose=1,
        env_id="trading_position",
        render_mode="logs",
        df=None,
        manager=dict(
            shared_memory=False,
            episode_num=float('inf'),
            max_retry=5,
            step_timeout=None,
            auto_reset=True,
            reset_timeout=None,
            retry_type='reset',
            retry_waiting_time=0.1,
            copy_on_get=True,
            context='fork',
            wait_num=float('inf'),
            step_wait_timeout=None,
            connect_timeout=60,
            reset_inplace=False,
            cfg_type='SyncSubprocessEnvManagerDict',
            type='subprocess',
        )
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=priority,
        priority_IS_weight=False,
        model=dict(
            obs_shape=1 * 26,
            action_shape=3,
            hidden_size=1024,
            encoder_hidden_size_list=[128, 512, 1024],
            #gru_bias=0.2,
            #gru_gating = True,
            memory_len=60 * 2,
            dropout=0.1,
            att_head_num=8 * 2,
            att_layer_num=3 * 2,
            att_head_dim=16 * 2,
            # att_mlp_num=2,
            # dueling = True,
            # encoder_hidden_size_list = [512, 512, 512],
        ),
        discount_factor=0.99,
        nstep=16 * 4,
        unroll_len=32 * 4,
        seq_len=32 * 4,
        learn=dict(
            update_per_collect=6,
            batch_size=128 * 2,
            learning_rate=0.0005,
            target_update_theta=0.001,
            value_rescale=True,
        ),
        collect=dict(
            # NOTE: It is important that set key traj_len_inf=True here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In R2D2 policy, for each collect_env, we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            n_sample=64,
            traj_len_inf=True,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=1440, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.99,
                end=0.1,
                decay=500000,
            ),
            replay_buffer=dict(
                replay_buffer_size=100000,
                # priority=priority,
                # priority_IS_weight=priority_IS_weight,
                # priority_power_factor=0.6,
                # IS_weight_power_factor=0.4,
                # IS_weight_anneal_train_iter=1e5,
                # max_use=float("inf"),
                # priority_max_limit=1000,
                # train_iter_per_log=100,
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.5,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.5,
            )
        ),
    ),
)
trading_position_r2d2_gtrxl_config = EasyDict(trading_position_r2d2_gtrxl_config)
main_config = trading_position_r2d2_gtrxl_config
trading_position_r2d2_gtrxl_create_config = dict(
    env=dict(
        type='trading_position',
        import_names=['envs.trading_position.trading_position_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d2_gtrxlc'),
)
trading_position_r2d2_gtrxl_create_config = EasyDict(trading_position_r2d2_gtrxl_create_config)
create_config = trading_position_r2d2_gtrxl_create_config
'''
# if you want to load a checkpoint, just enable this. After running a training, you always start from this ckpt.
trading_position_r2d2_gtrxl_config['hook'] = {
    'load_ckpt_before_run': 'path_to_checkpoint_file', #load ckp
    'save_ckpt_after_run': True,
}
'''
if __name__ == "__main__":
    # or you can enter `ding -m serial -c trading_position_r2d2_config.py -s 0`
    from ding.entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
