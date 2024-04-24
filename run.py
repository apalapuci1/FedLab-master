import sys
sys.path.append("../")

# configuration
from munch import Munch
from fedlab.models.mlp import MLP

model = MLP(784, 10)
args = Munch

args.total_client = 100
args.alpha = 0.5
args.seed = 42
args.preprocess = True
args.cuda = False
# prepare dataset
from torchvision import transforms
from fedlab.contrib.dataset.partitioned_mnist import PartitionedMNIST

fed_mnist = PartitionedMNIST(root="../datasets/mnist/",
                         path="../datasets/mnist/fedmnist/",
                         num_clients=args.total_client,
                         partition="noniid-labeldir",
                         dir_alpha=args.alpha,
                         seed=args.seed,
                         preprocess=args.preprocess,
                         download=True,
                         verbose=True,
                         transform=transforms.Compose(
                             [transforms.ToPILImage(), transforms.ToTensor()]))

dataset = fed_mnist.get_dataset(0) # get the 0-th client's dataset
dataloader = fed_mnist.get_dataloader(0, batch_size=128) # get the 0-th client's dataset loader with batch size 128

# client
from src.imlclienttrainer import SGDClientTrainer

# local train configuration
args.epochs = 5
args.batch_size = 128
args.lr = 0.1

trainer = SGDClientTrainer(model,  cuda=args.cuda) # serial trainer
# trainer = SGDClientTrainer(model, cuda=True) # single trainer

trainer.setup_dataset(fed_mnist)
trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# server
from src.imlServerHandler import SyncServerHandler

# global configuration
args.com_round = 10
args.sample_ratio = 0.1

handler = SyncServerHandler(model=model, global_round=args.com_round, sample_ratio=args.sample_ratio, cuda=args.cuda)

from fedlab.core import DistNetwork
from fedlab.core.client.manager import PassiveClientManager

# Client side. Put your trainer into a network manager.

args.ip = "127.0.0.1"
args.port = 3002
args.rank = 1
args.world_size = 2 # world_size = the number of client manager + 1 (server)

args.ethernet = None

client_network = DistNetwork(
    address=(args.ip, args.port),
    world_size=args.world_size,
    rank=args.rank,
    ethernet=args.ethernet,
)

# trainer can be ordinary trainer or serial trainer.
client_manager = PassiveClientManager(trainer=trainer,
                                network=client_network)

# Server side. Put your handler into a network manager.
from fedlab.core.server import SynchronousServerManager

server_network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=0, # the rank of server is 0 as default
                      ethernet=args.ethernet)

server_manager = SynchronousServerManager(handler=handler,
                                    network=server_network,
                                    mode="GLOBAL")