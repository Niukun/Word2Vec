package niukun;

import java.util.Collection;
import java.util.List;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

public class LoadModel_02 {

	public static void main(String[] args) {
//		Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("G:/学习工具/计算机/NLP/word2vec/bin/GoogleNews-vectors-negative300.bin.gz");
		Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("C:/D/NLPIR/paper/files/merge/Normal/NormalVector.txt");
		WeightLookupTable weightLookupTable = word2Vec.lookupTable();

		System.out.println(word2Vec.similarity("法国", "巴黎"));
		System.out.println(word2Vec.similarity("美国", "纽约"));
		System.out.println(word2Vec.similarity("男人", "女人"));
		System.out.println(word2Vec.similarity("主席", "胡锦涛"));
		System.out.println(word2Vec.similarity("枪支", "弹药"));
		Collection<String> lst = word2Vec.wordsNearest("朱元璋", 10);
        System.out.println(lst);
        lst = word2Vec.wordsNearest("北京", 10);
        System.out.println(lst);
        lst = word2Vec.wordsNearest("钱学森", 10);
        System.out.println(lst);
		
	}

}
