from train import Trainer
from myparser import parameter_parser
from utils.utils import tab_printer

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    trainer = Trainer(args)
    accuracy = trainer.train()
    print(f"accuracy:{accuracy}")
