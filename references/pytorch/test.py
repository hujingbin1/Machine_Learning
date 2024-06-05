from text2vec import SentenceModel
from sentence_transformers.util import cos_sim
from uniem.finetuner import FineTuner
import pandas as pd


class Similar:
    def __init__(self, text2vec_model="moka-ai/m3e-base"):
        self.model_path = text2vec_model
        self.text2vec_model = SentenceModel(self.model_path)

    def get_similar(self, sentence1: str, sentence2: str):
        embedding1 = self.text2vec_model.encode(sentence1)
        embedding2 = self.text2vec_model.encode(sentence2)
        return cos_sim(embedding1, embedding2)

    def finetune_model(
        self, input_file: str, output_dir: str, epochs: int, batch_size: int
    ):
        df = pd.read_excel(input_file)
        finetuner = FineTuner.from_pretrained(
            self.model_path, dataset=df.to_dict("records")
        )
        fintuned_model = finetuner.run(
            epochs=epochs, output_dir=output_dir, batch_size=batch_size
        )


if __name__ == "__main__":
    # 选择对应型号的 bge 模型
    similar = Similar("moka-ai/m3e-base")

    # 使用源模型计算文本相似度
    similarity_score = similar.get_similar(
        "谜底：白羊", "猜谜语：一身卷卷细毛，吃的青青野草，过了数九寒冬，无私献出白毛。 （打一动物）"
    )
    print(similarity_score)

    # 微调源模型（根据实际情况修改对应参数）
    similar.finetune_model(
        input_file="data.xlsx", output_dir="finetuned-model", epochs=1, batch_size=64
    )

    # 使用微调后的模型计算文本相似度
    finetune_similar = Similar(f"finetuned-model/model")
    similarity_score = finetune_similar.get_similar(
        "谜底：白羊", "猜谜语：一身卷卷细毛，吃的青青野草，过了数九寒冬，无私献出白毛。 （打一动物）"
    )
    print(similarity_score) 作者：AI日日新 https://www.bilibili.com/read/cv26440355/ 出处：bilibili