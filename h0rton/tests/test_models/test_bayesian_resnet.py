import unittest
import numpy as np
import torch
import h0rton.models as models

class TestBayesianResNet(unittest.TestCase):
    """A suite of tests on ResNet models
    
    """

    @classmethod
    def setUpClass(cls):
        cls.Y_dim = 12
        cls.out_dim = cls.Y_dim**2 + 3*cls.Y_dim + 1
        cls.dropout_rate = 0.05
        cls.batch_size = 7
        cls.dummy_X = torch.randn(cls.batch_size, 1, 64, 64)
        cls.dummy_Y = torch.randn(cls.batch_size, cls.out_dim)

    def test_bayesian_resnet_forward(self):
        """Test instantiation and forward method of BayesianResNet

        """
        resnet34 = models.resnet34(num_classes=self.out_dim, dropout_rate=self.dropout_rate)
        pred34 = resnet34.forward(self.dummy_X)
        resnet44 = models.resnet44(num_classes=self.out_dim, dropout_rate=self.dropout_rate)
        pred44 = resnet44.forward(self.dummy_X)
        resnet56 = models.resnet56(num_classes=self.out_dim, dropout_rate=self.dropout_rate)
        pred56 = resnet56.forward(self.dummy_X)
        resnet50 = models.resnet50(num_classes=self.out_dim, dropout_rate=self.dropout_rate)
        pred50 = resnet50.forward(self.dummy_X)
        resnet101 = models.resnet101(num_classes=self.out_dim, dropout_rate=self.dropout_rate)
        pred101 = resnet101.forward(self.dummy_X)
        np.testing.assert_array_equal(pred34.shape, [self.batch_size, self.out_dim], err_msg="output shape of resnet34")
        np.testing.assert_array_equal(pred44.shape, [self.batch_size, self.out_dim], err_msg="output shape of resnet44")
        np.testing.assert_array_equal(pred56.shape, [self.batch_size, self.out_dim], err_msg="output shape of resnet56")
        np.testing.assert_array_equal(pred50.shape, [self.batch_size, self.out_dim], err_msg="output shape of resnet50")
        np.testing.assert_array_equal(pred101.shape, [self.batch_size, self.out_dim], err_msg="output shape of resnet101")
        
    def test_bayesian_resnet_dropout(self):
        """Test if dropout rate is propagated to basic blocks

        """
        resnet44 = models.resnet44(num_classes=self.out_dim, dropout_rate=self.dropout_rate)
        # Take the first BasicBlock in each layer
        assert resnet44.layer1[0].dropout_rate == self.dropout_rate
        assert resnet44.layer2[0].dropout_rate == self.dropout_rate
        assert resnet44.layer3[0].dropout_rate == self.dropout_rate

    def test_activation_maps_layer3(self):
        """Test if the 3-layer BNN has the correctly shaped activation maps (intermediate feature maps)

        """
        bnn = models.resnet44(num_classes=self.out_dim, dropout_rate=self.dropout_rate)
        activations = bnn._forward_debug(self.dummy_X)
        np.testing.assert_array_equal(activations[0], self.dummy_X.shape, err_msg="input")
        np.testing.assert_array_equal(activations[1], [self.batch_size, 64, 32, 32], err_msg="after conv1 with stride 2")
        np.testing.assert_array_equal(activations[2], [self.batch_size, 64, 16, 16], err_msg="after maxpool")
        np.testing.assert_array_equal(activations[3], [self.batch_size, 64, 16, 16], err_msg="after layer1")
        np.testing.assert_array_equal(activations[4], [self.batch_size, 128, 8, 8], err_msg="after layer2")
        np.testing.assert_array_equal(activations[5], [self.batch_size, 256, 4, 4], err_msg="after layer3")
        np.testing.assert_array_equal(activations[6], [self.batch_size, 256, 4, 4], err_msg="after layer4")
        np.testing.assert_array_equal(activations[7], [self.batch_size, 256, 1, 1], err_msg="after avgpool")
        np.testing.assert_array_equal(activations[-1], [self.batch_size, self.out_dim], err_msg="output")
        
    def test_activation_maps_layer4(self):
        """Test if the 4-layer BNN has the correctly shaped activation maps

        """
        bnn = models.resnet34(num_classes=self.out_dim, dropout_rate=self.dropout_rate)
        activations = bnn._forward_debug(self.dummy_X)
        np.testing.assert_array_equal(activations[0], self.dummy_X.shape, err_msg="input")
        np.testing.assert_array_equal(activations[1], [self.batch_size, 64, 32, 32], err_msg="after conv1 with stride 2")
        np.testing.assert_array_equal(activations[2], [self.batch_size, 64, 16, 16], err_msg="after maxpool")
        np.testing.assert_array_equal(activations[3], [self.batch_size, 64, 16, 16], err_msg="after layer1")
        np.testing.assert_array_equal(activations[4], [self.batch_size, 128, 8, 8], err_msg="after layer2")
        np.testing.assert_array_equal(activations[5], [self.batch_size, 256, 4, 4], err_msg="after layer3")
        np.testing.assert_array_equal(activations[6], [self.batch_size, 512, 2, 2], err_msg="after layer4")
        np.testing.assert_array_equal(activations[7], [self.batch_size, 512, 1, 1], err_msg="after avgpool")
        np.testing.assert_array_equal(activations[-1], [self.batch_size, self.out_dim], err_msg="output")

if __name__ == '__main__':
    unittest.main()