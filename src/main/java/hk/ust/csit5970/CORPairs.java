package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.commons.cli.Options;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Compute the bigram count using "pairs" approach
 */
public class CORPairs extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(CORPairs.class);
	private static final String TEMP_OUTPUT = "temp_output";

	/*
	 * First-pass Mapper: counts word frequencies
	 */
	private static class CORMapper1 extends
			Mapper<LongWritable, Text, Text, IntWritable> {
		private static final Text WORD = new Text();
		private static final IntWritable COUNT = new IntWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			HashMap<String, Integer> word_set = new HashMap<String, Integer>();
			// Please use this tokenizer! DO NOT implement a tokenizer by yourself!
			String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);
			
			// Count each word in the document
			while (doc_tokenizer.hasMoreTokens()) {
				String word = doc_tokenizer.nextToken().toLowerCase();
				if (word_set.containsKey(word)) {
					word_set.put(word, word_set.get(word) + 1);
				} else {
					word_set.put(word, 1);
				}
			}
			
			// Emit each word with its count
			for (Map.Entry<String, Integer> entry : word_set.entrySet()) {
				WORD.set(entry.getKey());
				COUNT.set(entry.getValue());
				context.write(WORD, COUNT);
			}
		}
	}

	/*
	 * First-pass reducer: aggregates word counts
	 */
	private static class CORReducer1 extends
			Reducer<Text, IntWritable, Text, IntWritable> {
		private static final IntWritable SUM = new IntWritable();
		
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable value : values) {
				sum += value.get();
			}
			SUM.set(sum);
			context.write(key, SUM);
		}
	}


	/*
	 * Second-pass Mapper: emits word pairs for all co-occurring words
	 */
	public static class CORPairsMapper2 extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
		private static final IntWritable ONE = new IntWritable(1);
		private static final PairOfStrings PAIR = new PairOfStrings();
		
		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			// Please use this tokenizer! DO NOT implement a tokenizer by yourself!
			String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);
			
			// Find unique words in the line
			Set<String> uniqueWords = new HashSet<String>();
			while (doc_tokenizer.hasMoreTokens()) {
				uniqueWords.add(doc_tokenizer.nextToken().toLowerCase());
			}
			
			// Skip if only one or zero words
			if (uniqueWords.size() <= 1) {
				return;
			}
			
			// Create a sorted list of unique words
			List<String> sortedWords = new ArrayList<String>(uniqueWords);
			Collections.sort(sortedWords);
			
			// Emit each word pair (in alphabetical order)
			for (int i = 0; i < sortedWords.size(); i++) {
				for (int j = i + 1; j < sortedWords.size(); j++) {
					String word1 = sortedWords.get(i);
					String word2 = sortedWords.get(j);
					
					// Ensure word1 is alphabetically smaller than word2
					PAIR.set(word1, word2);
					context.write(PAIR, ONE);
				}
			}
		}
	}

	/*
	 * Second-pass Combiner: combines counts for the same word pair
	 */
	private static class CORPairsCombiner2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
		private static final IntWritable SUM = new IntWritable();
		
		@Override
		protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable value : values) {
				sum += value.get();
			}
			SUM.set(sum);
			context.write(key, SUM);
		}
	}

	/*
	 * Second-pass Reducer: calculates correlation coefficients
	 */
	public static class CORPairsReducer2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
		private final static Map<String, Integer> word_total_map = new HashMap<String, Integer>();
		private static final DoubleWritable CORRELATION = new DoubleWritable();

		/*
		 * Preload the middle result file.
		 * In the middle result file, each line contains a word and its frequency Freq(A), seperated by "\t"
		 */
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			Path middle_result_path = new Path("mid/part-r-00000");
			Configuration middle_conf = new Configuration();
			try {
				FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);

				if (!fs.exists(middle_result_path)) {
					throw new IOException(middle_result_path.toString() + "not exist!");
				}

				FSDataInputStream in = fs.open(middle_result_path);
				InputStreamReader inStream = new InputStreamReader(in);
				BufferedReader reader = new BufferedReader(inStream);

				LOG.info("reading...");
				String line = reader.readLine();
				String[] line_terms;
				while (line != null) {
					line_terms = line.split("\t");
					word_total_map.put(line_terms[0], Integer.valueOf(line_terms[1]));
					LOG.info("read one line!");
					line = reader.readLine();
				}
				reader.close();
				LOG.info("finished！");
			} catch (Exception e) {
				System.out.println(e.getMessage());
			}
		}

		@Override
		protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			String word1 = key.getLeftElement();
			String word2 = key.getRightElement();
			
			// Skip if we don't have frequencies for both words
			if (!word_total_map.containsKey(word1) || !word_total_map.containsKey(word2)) {
				return;
			}
			
			// Sum up the pair frequency
			int pairFreq = 0;
			for (IntWritable value : values) {
				pairFreq += value.get();
			}
			
			// Get individual word frequencies
			int freq1 = word_total_map.get(word1);
			int freq2 = word_total_map.get(word2);
			
			// Calculate correlation: COR(A, B) = Freq(A, B) / (Freq(A) * Freq(B))
			double correlation = (double) pairFreq / (freq1 * freq2);
			
			CORRELATION.set(correlation);
			context.write(key, CORRELATION);
		}
	}

	private static final class MyPartitioner extends Partitioner<PairOfStrings, FloatWritable> {
		@Override
		public int getPartition(PairOfStrings key, FloatWritable value, int numReduceTasks) {
			return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public CORPairs() {
	}

	private static final String INPUT = "input";
	private static final String MIDDLE = "middle";
	private static final String OUTPUT = "output";
	private static final String NUM_REDUCERS = "numReducers";

	/**
	 * Runs this tool.
	 */
	@SuppressWarnings({ "static-access" })
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("input path").create(INPUT));
		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("output path").create(OUTPUT));
		options.addOption(OptionBuilder.withArgName("num").hasArg()
				.withDescription("number of reducers").create(NUM_REDUCERS));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: "
					+ exp.getMessage());
			return -1;
		}

		// Lack of arguments
		if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}

		String inputPath = cmdline.getOptionValue(INPUT);
		String middlePath = "mid";
		String outputPath = cmdline.getOptionValue(OUTPUT);

		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
				.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

		LOG.info("Tool: " + CORPairs.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Setup for the first-pass MapReduce
		Configuration conf1 = new Configuration();

		Job job1 = Job.getInstance(conf1, "Firstpass");

		job1.setJarByClass(CORPairs.class);
		job1.setMapperClass(CORMapper1.class);
		job1.setReducerClass(CORReducer1.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);

		FileInputFormat.setInputPaths(job1, new Path(inputPath));
		FileOutputFormat.setOutputPath(job1, new Path(middlePath));

		// Delete the output directory if it exists already.
		Path middleDir = new Path(middlePath);
		FileSystem.get(conf1).delete(middleDir, true);

		// Time the program
		long startTime = System.currentTimeMillis();
		job1.waitForCompletion(true);
		LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		// Setup for the second-pass MapReduce

		// Delete the output directory if it exists already.
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf1).delete(outputDir, true);


		Configuration conf2 = new Configuration();
		Job job2 = Job.getInstance(conf2, "Secondpass");

		job2.setJarByClass(CORPairs.class);
		job2.setMapperClass(CORPairsMapper2.class);
		job2.setCombinerClass(CORPairsCombiner2.class);
		job2.setReducerClass(CORPairsReducer2.class);

		job2.setOutputKeyClass(PairOfStrings.class);
		job2.setOutputValueClass(DoubleWritable.class);
		job2.setMapOutputValueClass(IntWritable.class);
		job2.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job2, new Path(inputPath));
		FileOutputFormat.setOutputPath(job2, new Path(outputPath));

		// Time the program
		startTime = System.currentTimeMillis();
		job2.waitForCompletion(true);
		LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new CORPairs(), args);
	}
}