# rl-config
from easydict import EasyDict

collector_env_num = 1
evaluator_env_num = 1
trading_position_r2d2_config = dict(
    exp_name='trading_position_r2d2_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,

        positions=[-1, 0, 1],
        # Prev number of kline will keep obs and LSTM NN will use it for choose the action of next step
        windows=60 * 6,
        trading_fees=0.0004,
        borrow_interest_rate=0,
        portfolio_initial_value=1000000,
        initial_position='random',
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
        priority=True,
        priority_IS_weight=False,
        discount_factor=0.99,
        nstep=120,

        burnin_step=60,

        learn_unroll_len=180,
        model=dict(
            # window_size x obs features = 20 x 9 = 180 (This shape is used for RNN and input shape of Conv2d).
            obs_shape=60 * 6 * 26,
            action_shape=3,
            # Used for output of Linear layer.
            encoder_hidden_size_list=[1024, 1024, 1024]
        ),
        learn=dict(
            update_per_collect=60,
            batch_size=512,
            learning_rate=1E-4,
            target_update_freq=1500,
            iqn=True,
        ),
        collect=dict(
            n_sample=256, 
            unroll_len= 180 + 60, #learn_unroll + burn
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=1440, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.99,
                end=0.3,
                decay=1000000,
            ), replay_buffer=dict(replay_buffer_size=1000000, )
        ),
    ),
)
trading_position_r2d2_config = EasyDict(trading_position_r2d2_config)
main_config = trading_position_r2d2_config
trading_position_r2d2_create_config = dict(
    env=dict(
        type='trading_position',
        import_names=['envs.trading_position.trading_position_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d2'),
)
trading_position_r2d2_create_config = EasyDict(trading_position_r2d2_create_config)
create_config = trading_position_r2d2_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c trading_position_r2d2_config.py -s 0`
    from ding.entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
