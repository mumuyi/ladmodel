package cn.nuaa.ai.main;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import cn.nuaa.ai.com.FileUtil;
import cn.nuaa.ai.conf.ConstantConfig;
import cn.nuaa.ai.conf.PathConfig;

/**Liu Yang's implementation of Gibbs Sampling of LDA
 * @author yangliu
 * @blog http://blog.csdn.net/yangliuy
 * @mail yangliuyx@gmail.com
 */

public class LdaGibbsSampling {
	
	public static class modelparameters {
		public float alpha = 0.5f; //usual value is 50 / K
		public float beta = 0.1f;//usual value is 0.1
		public int topicNum = 100;
		public int iteration = 100;
		public int saveStep = 10;
		public int beginSaveIters = 50;
	}
	
	/**Get parameters from configuring file. If the 
	 * configuring file has value in it, use the value.
	 * Else the default value in program will be used
	 * @param ldaparameters
	 * @param parameterFile
	 * @return void
	 */
	public static void getParametersFromFile(modelparameters ldaparameters,
			String parameterFile) {
		// TODO Auto-generated method stub
		ArrayList<String> paramLines = new ArrayList<String>();
		FileUtil.readLines(parameterFile, paramLines);
		for(String line : paramLines){
			String[] lineParts = line.split("\t");
			switch(parameters.valueOf(lineParts[0])){
			case alpha:
				ldaparameters.alpha = Float.valueOf(lineParts[1]);
				break;
			case beta:
				ldaparameters.beta = Float.valueOf(lineParts[1]);
				break;
			case topicNum:
				ldaparameters.topicNum = Integer.valueOf(lineParts[1]);
				break;
			case iteration:
				ldaparameters.iteration = Integer.valueOf(lineParts[1]);
				break;
			case saveStep:
				ldaparameters.saveStep = Integer.valueOf(lineParts[1]);
				break;
			case beginSaveIters:
				ldaparameters.beginSaveIters = Integer.valueOf(lineParts[1]);
				break;
			}
		}
	}
	
	public enum parameters{
		alpha, beta, topicNum, iteration, saveStep, beginSaveIters;
	}
	
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		//LDA 原文件路径;
		String originalDocsPath = PathConfig.ldaDocsPath;
		//LDA 结果保存路径;
		String resultPath = PathConfig.LdaResultsPath;
		//LDA 参数文件路径;
		String parameterFile= ConstantConfig.LDAPARAMETERFILE;
		
		//获取LDA 参数;
		modelparameters ldaparameters = new modelparameters();
		getParametersFromFile(ldaparameters, parameterFile);
		
		//获取LDA 原文件;
		Documents docSet = new Documents();
		docSet.readDocs(originalDocsPath);
		System.out.println("wordMap size " + docSet.termToIndexMap.size());
		
		//建立LDA 结果文件;
		FileUtil.mkdir(new File(resultPath));
		
		//实例化LDA model;
		LdaModel model = new LdaModel(ldaparameters);
		//初始化;
		System.out.println("1 Initialize the model ...");
		model.initializeModel(docSet);
		//学习;
		System.out.println("2 Learning and Saving the model ...");
		model.inferenceModel(docSet);
		//输出;
		System.out.println("3 Output the final model ...");
		model.saveIteratedModel(ldaparameters.iteration, docSet);
		System.out.println("Done!");
	}
}
