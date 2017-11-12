package com.iitp.flyPig;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.*;

public class Preprocess_fnc {
	private String input = "data/fnc/train_stances.txt";
	private String output = "data/fnc/processed.csv";

	private List<String> stopWord = new ArrayList<String>();
	int totalTitleWordCount = 0;
	int totalBodyWordCount = 0;
	int lNum = 0;
	Map<String, String> map = new HashMap<>();

	public void mapping() {
		try {
			BufferedReader br = new BufferedReader(new FileReader("data/fnc/train_bodies.txt"));
			String line = "";
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.isEmpty())
					continue;
				String[] cols = line.split("\t");

				map.put(cols[0].toLowerCase(), cols[1].toLowerCase());

			}
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void stopWord() {
		try {
			BufferedReader br = new BufferedReader(new FileReader("data/stopwords_en.txt"));
			String line = "";
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.isEmpty())
					continue;
				System.out.println(line);
				stopWord.add(line.toLowerCase());

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
			int lineNum = 0;
			while ((line = br.readLine()) != null) {
				// universesal label
				lineNum++;
				line = line.trim();
				String[] cols = line.split("\t");
				// removing hyper link
				line = line.replaceAll("https?://\\S+\\s?", "");
				// removing non-ascii
				line = line.replaceAll("[^\\x00-\\x7F]", "");
				// removing punctuation
				line = line.replaceAll("[^a-zA-Z0-9]", "");
				String[] titleWords = cols[0].split("\\s+");
				if(!map.containsKey(cols[1]))
					 continue;
				String[] bodyWords = map.get(cols[1].trim()).split("\\s+");
				String title = "";
				String body = "";
				int titleWordCount = 0;
				int bodyWordCount = 0;
				for (String str : titleWords) {
					str = str.replaceAll("[^a-zA-Z0-9]", "");
					// removing non-ascii
					str = str.replaceAll("[^\\x00-\\x7F]", "");
					// removing hyper link
					str = str.replaceAll("https?://\\S+\\s?", "");
					if (!stopWord.contains(str.trim().toLowerCase())) {
						title += str + " ";
						titleWordCount++;
					}
				}
				
				for (String str : bodyWords) {
					str = str.replaceAll("[^a-zA-Z0-9]", "");
					// removing non-ascii
					str = str.replaceAll("[^\\x00-\\x7F]", "");
					// removing hyper link
					str = str.replaceAll("https?://\\S+\\s?", "");
					if (!stopWord.contains(str.trim().toLowerCase())) {
						body += str + " ";
						bodyWordCount++;
					}
				}
				if (title.trim().length() == 0 || body.trim().length() == 0)
					continue;
				if (totalBodyWordCount < bodyWordCount) {
					totalBodyWordCount = bodyWordCount;
					lNum = lineNum;
				}
				if (totalTitleWordCount < titleWordCount)
					totalTitleWordCount = titleWordCount;

				bw.write(title + "\t" + body + "\t" + cols[2]);
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
		Preprocess_fnc pp = new Preprocess_fnc();
		pp.mapping();
		pp.stopWord();
		pp.run();
		System.out.println(pp.totalBodyWordCount + "\n" + pp.totalTitleWordCount + "\n" + pp.lNum);
		System.out.println("Done!..........");

	}

}
