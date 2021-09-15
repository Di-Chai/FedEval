from unittest import TestCase

from .configuration import (_D_PARTITION_KEY, _DEFAULT_D_CFG, _DEFAULT_MDL_CFG,
                            _DEFAULT_RT_CFG, ConfigurationManager, _DataConfig)
from .role import Role


class ConfigurationManagerTestCase(TestCase):
    def setUp(self):
        self.cfg_mgr = ConfigurationManager()

    def test_default_cfgs(self):
        self.assertDictEqual(self.cfg_mgr.data_config.inner, _DEFAULT_D_CFG)
        self.assertDictEqual(self.cfg_mgr.model_config.inner, _DEFAULT_MDL_CFG)
        self.assertDictEqual(self.cfg_mgr.runtime_config.inner, _DEFAULT_RT_CFG)

    def test_cfg_write_availability(self):
        def set_data_cfg():
            self.cfg_mgr.data_config = {}
        def set_model_cfg():
            self.cfg_mgr.model_config = {}
        def set_runtime_cfg():
            self.cfg_mgr.runtime_config = {}
        self.assertRaises(AttributeError, set_data_cfg)
        self.assertRaises(AttributeError, set_model_cfg)
        self.assertRaises(AttributeError, set_runtime_cfg)

    def test_rt_cfg_accessability(self):
        _ = self.cfg_mgr.model_config.ml_config
        _ = self.cfg_mgr.model_config.strategy_config

    def test_filename_setters(self):
        invalid_names = ['/data_config.yml', 'data\\_config.yml']
        for invalid_name in invalid_names:
            with self.assertRaisesRegex(ValueError, 'sep'):
                self.cfg_mgr.data_config_filename = invalid_name
        self.cfg_mgr.data_config_filename = 'data_config.yml'

    def test_role_setting(self):
        ano_mgr = ConfigurationManager()
        ano_mgr.role = Role.Server
        self.assertEqual(self.cfg_mgr.role, Role.Server)


class DataConfigPartitionTestCase(TestCase):
    def setUp(self) -> None:
        self.rcfg = _DEFAULT_D_CFG.copy()

    def test_sum_limit(self):
        self.rcfg[_D_PARTITION_KEY] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "small"):
            _DataConfig(self.rcfg)

    def test_no_neg(self):
        self.rcfg[_D_PARTITION_KEY] = [-1, 2, 0]
        with self.assertRaisesRegex(ValueError, 'negetive'):
            _DataConfig(self.rcfg)

    def test_not_enough(self):
        self.rcfg[_D_PARTITION_KEY] = [1, 2]
        with self.assertRaisesRegex(ValueError, '3'):
            _DataConfig(self.rcfg)

    def test_too_many(self):
        self.rcfg[_D_PARTITION_KEY] = [1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, '3'):
            _DataConfig(self.rcfg)

    def test_copy_attribute(self):
        self.rcfg[_D_PARTITION_KEY] = [0.1, 0.1, 0.8]
        d_cfg = _DataConfig(self.rcfg)
        self.assertIsNot(d_cfg.data_partition, self.rcfg[_D_PARTITION_KEY])

    def test_ok(self):
        partition = [0.1, 2, 3]
        summation = sum(partition)
        partition_normalized = [i / summation for i in partition]
        self.rcfg[_D_PARTITION_KEY] = partition
        d_cfg = _DataConfig(self.rcfg)
        _data_partition = d_cfg.data_partition
        self.assertIsNot(_data_partition, partition)
        self.assertAlmostEqual(sum(_data_partition), 1.0)
        for i in range(3):
            self.assertAlmostEqual(partition_normalized[i], _data_partition[i])


class DataConfigTestCase(TestCase):
    def setUp(self):
        self.cfg = _DataConfig(_DEFAULT_D_CFG)

    def test_inner_copy(self):
        inner = self.cfg.inner
        self.assertDictEqual(inner, _DEFAULT_D_CFG)
        self.assertIsNot(inner, _DEFAULT_D_CFG)

    def test_sample_size(self):
        self.assertGreater(self.cfg.sample_size, 0)
        self.assertTrue(isinstance(self.cfg.sample_size, int))

    def test_path_sep_in_default_dir_name(self):
        possible_seps = ['/', '\\', '.']
        for sep in possible_seps:
            self.assertFalse(sep in self.cfg.dir_name)

# TODO(fgh) add tests for config conversions 
