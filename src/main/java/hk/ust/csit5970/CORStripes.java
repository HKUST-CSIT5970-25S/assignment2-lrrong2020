package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

/**
 * Compute the bigram count using "pairs" approach
 */
public class CORStripes extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(CORStripes.class);
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
	 * Second-pass Mapper: emits word stripes for all co-occurring words
	 */
	public static class CORStripesMapper2 extends Mapper<LongWritable, Text, Text, MapWritable> {
		private static final Text WORD = new Text();
		
		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			Set<String> sorted_word_set = new TreeSet<String>();
			// Please use this tokenizer! DO NOT implement a tokenizer by yourself!
			String doc_clean = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer doc_tokenizers = new StringTokenizer(doc_clean);
			
			// Extract unique words sorted alphabetically
			while (doc_tokenizers.hasMoreTokens()) {
				sorted_word_set.add(doc_tokenizers.nextToken().toLowerCase());
			}
			
			// Skip if only one or zero words
			if (sorted_word_set.size() <= 1) {
				return;
			}
			
			// Convert to array for easier indexing
			String[] words = sorted_word_set.toArray(new String[sorted_word_set.size()]);
			
			// For each word, emit a stripe with all co-occurring words
			for (int i = 0; i < words.length; i++) {
				MapWritable stripe = new MapWritable();
				for (int j = i + 1; j < words.length; j++) {
					// Only emit pairs where the left word is alphabetically smaller
					Text coWord = new Text(words[j]);
					IntWritable ONE = new IntWritable(1);
					stripe.put(coWord, ONE);
				}
				
				if (!stripe.isEmpty()) {
					WORD.set(words[i]);
					context.write(WORD, stripe);
				}
			}
		}
	}

	/*
	 * Second-pass Combiner: combines stripes with the same key
	 */
	public static class CORStripesCombiner2 extends Reducer<Text, MapWritable, Text, MapWritable> {
		static IntWritable ZERO = new IntWritable(0);

		@Override
		protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
			MapWritable combined = new MapWritable();
			
			// Combine all stripes for this key
			for (MapWritable value : values) {
				for (Writable coWord : value.keySet()) {
					IntWritable count = (IntWritable) value.get(coWord);
					
					if (combined.containsKey(coWord)) {
						// Add to existing count
						IntWritable existingCount = (IntWritable) combined.get(coWord);
						existingCount.set(existingCount.get() + count.get());
						combined.put(coWord, existingCount);
					} else {
						// Create new count
						combined.put(coWord, new IntWritable(count.get()));
					}
				}
			}
			
			context.write(key, combined);
		}
	}

	/*
	 * Second-pass Reducer: calculates correlation coefficients
	 */
	public static class CORStripesReducer2 extends Reducer<Text, MapWritable, PairOfStrings, DoubleWritable> {
		private static Map<String, Integer> word_total_map = new HashMap<String, Integer>();
		private static IntWritable ZERO = new IntWritable(0);
		private static final PairOfStrings PAIR = new PairOfStrings();
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
		protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
			String word1 = key.toString();
			// Skip if we don't have frequency for this word
			if (!word_total_map.containsKey(word1)) {
				return;
			}
			
			int freq1 = word_total_map.get(word1);
			
			// Combine all stripes for this key
			MapWritable combined = new MapWritable();
			for (MapWritable value : values) {
				for (Writable coWord : value.keySet()) {
					IntWritable count = (IntWritable) value.get(coWord);
					
					if (combined.containsKey(coWord)) {
						// Add to existing count
						IntWritable existingCount = (IntWritable) combined.get(coWord);
						existingCount.set(existingCount.get() + count.get());
						combined.put(coWord, existingCount);
					} else {
						// Create new count
						combined.put(coWord, new IntWritable(count.get()));
					}
				}
			}
			
			// Calculate correlation for each co-occurring word
			for (Writable coWordObj : combined.keySet()) {
				String word2 = ((Text) coWordObj).toString();
				
				// Skip if we don't have frequency for co-word
				if (!word_total_map.containsKey(word2)) {
					continue;
				}
				
				int freq2 = word_total_map.get(word2);
				int pairFreq = ((IntWritable) combined.get(coWordObj)).get();
				
				// Calculate correlation: COR(A, B) = Freq(A, B) / (Freq(A) * Freq(B))
				double correlation = (double) pairFreq / (freq1 * freq2);
				
				// Ensure word1 is alphabetically smaller than word2
				PAIR.set(word1, word2);
				CORRELATION.set(correlation);
				context.write(PAIR, CORRELATION);
			}
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public CORStripes() {
	}

	private static final String INPUT = "input";
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

		LOG.info("Tool: " + CORStripes.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - middle path: " + middlePath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Setup for the first-pass MapReduce
		Configuration conf1 = new Configuration();

		Job job1 = Job.getInstance(conf1, "Firstpass");

		job1.setJarByClass(CORStripes.class);
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

		job2.setJarByClass(CORStripes.class);
		job2.setMapperClass(CORStripesMapper2.class);
		job2.setCombinerClass(CORStripesCombiner2.class);
		job2.setReducerClass(CORStripesReducer2.class);

		job2.setOutputKeyClass(PairOfStrings.class);
		job2.setOutputValueClass(DoubleWritable.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(MapWritable.class);
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
		ToolRunner.run(new CORStripes(), args);
	}
}