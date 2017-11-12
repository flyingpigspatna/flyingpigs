package com.iitp.flyPig;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Preprocessing {

	private String input = "data/total.txt";
	private String output = "data/processed.csv";

	private List<String> stopWord = new ArrayList<String>();
	int totalTitleWordCount=0;
	int totalBodyWordCount=0;
	int lNum=0;

	public void stopWord() {
		try {
			BufferedReader br = new BufferedReader(new FileReader("data/stopwords_en.txt"));
			String line = "";
			while ((line=br.readLine()) != null) {
				line = line.trim();
				if(line.isEmpty())
					continue;
				System.out.println(line);
				stopWord.add(line);

			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void run() {

		BufferedReader br = null;
		BufferedWriter bw = null;
		try {

			Reader reader = new InputStreamReader(new FileInputStream(input), "UTF-8");
			br = new BufferedReader(reader);
			Writer writer = new OutputStreamWriter(new FileOutputStream(output), "UTF-8");
			bw = new BufferedWriter(writer);

			String line = "";
			br.readLine();
			int lineNum=0;
			while ((line = br.readLine()) != null) {
				// universesal label
				lineNum++;
				line = line.trim();
				String[] cols = line.split("\t");
				if(cols[0].equalsIgnoreCase("nan"))
					continue;
				// removing hyper link
				line = line.replaceAll("https?://\\S+\\s?", "");
				// removing non-ascii
				line = line.replaceAll("[^\\x00-\\x7F]", "");
				// removing punctuation
				line = line.replaceAll("[^a-zA-Z0-9]", "");
				String []titleWords = cols[0].split("\\s+");
				String []bodyWords = cols[1].split("\\s+");
				String title ="";
				String body ="";
				int titleWordCount = 0;
				int bodyWordCount = 0 ;
				for(String str:titleWords) {
					str = str.replaceAll("[^a-zA-Z0-9]", "");
					// removing non-ascii
					str = str.replaceAll("[^\\x00-\\x7F]", "");
					// removing hyper link
					str = str.replaceAll("https?://\\S+\\s?", "");
					if(!stopWord.contains(str.trim())) {
						title +=str+" ";
						titleWordCount++;
					}
				}
				for(String str:bodyWords) {
					str = str.replaceAll("[^a-zA-Z0-9]", "");
					// removing non-ascii
					str = str.replaceAll("[^\\x00-\\x7F]", "");
					// removing hyper link
					str = str.replaceAll("https?://\\S+\\s?", "");
					if(!stopWord.contains(str.trim())) {
						body +=str+" ";
						bodyWordCount++;
					}
				}
				if(title.trim().length()==0||body.trim().length()==0)
					continue;
				if(totalBodyWordCount<bodyWordCount) {
					totalBodyWordCount =  bodyWordCount;
					lNum = lineNum;
				}
				if(totalTitleWordCount<titleWordCount)
					totalTitleWordCount =  titleWordCount;
			    int score = 0;
			    System.out.println(line+"\n"+lineNum);
			    try {
			    	if(Double.parseDouble(cols[2])>0.5) {
				    	score = 1;
				    	
			    	}
			    }catch(NumberFormatException e) {
			    	System.out.println(e);
			    	score = 0 ;
			    }
			    
			    bw.write(title+"\t"+body+"\t"+score);
				bw.write("\n");
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			br.close();
			br.close();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		Preprocessing pp=new Preprocessing();
		pp.stopWord();
		pp.run();
		System.out.println(pp.totalBodyWordCount+"\n"+pp.totalTitleWordCount+"\n"+pp.lNum);
		System.out.println("Done!..........");

	}

}
