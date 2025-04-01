package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute the bigram count using the "stripes" approach
 */
public class BigramFrequencyStripes extends Configured implements Tool {
	private static final Logger LOG = Logger
			.getLogger(BigramFrequencyStripes.class);

	/*
	 * Mapper: emits word -> stripe of (nextWord, 1) pairs
	 */
	private static class MyMapper extends
			Mapper<LongWritable, Text, Text, HashMapStringIntWritable> {
		private static final Text WORD = new Text();

		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String line = value.toString();
			String[] tokens = line.trim().toLowerCase().split("\\s+");

			if (tokens.length < 2) return;

			for (int i = 0; i < tokens.length - 1; i++) {
				String word = tokens[i];
				String nextWord = tokens[i + 1];

				// Emit current word -> (nextWord, 1)
				WORD.set(word);
				HashMapStringIntWritable stripes = new HashMapStringIntWritable();
				stripes.increment(nextWord, 1);

				// Also add a special count for the total
				stripes.increment("", 1);
				context.write(WORD, stripes);
			}

			// Don't forget the last word
			WORD.set(tokens[tokens.length - 1]);
			HashMapStringIntWritable stripes = new HashMapStringIntWritable();
			stripes.increment("", 1);
			context.write(WORD, stripes);
		}
	}

	/*
	 * Combiner: combines stripes with the same key
	 */
	private static class MyCombiner extends
			Reducer<Text, HashMapStringIntWritable, Text, HashMapStringIntWritable> {
		@Override
		public void reduce(Text key, Iterable<HashMapStringIntWritable> values, Context context)
				throws IOException, InterruptedException {
			HashMapStringIntWritable combined = new HashMapStringIntWritable();
			for (HashMapStringIntWritable value : values) {
				combined.plus(value);
			}
			context.write(key, combined);
		}
	}

	/*
	 * Reducer: calculate relative frequencies from stripes
	 */
	private static class MyReducer extends
			Reducer<Text, HashMapStringIntWritable, Text, Text> {
		private static final Text RESULT_KEY = new Text();
		private static final Text RESULT_VALUE = new Text();

		@Override
		public void reduce(Text key, Iterable<HashMapStringIntWritable> values, Context context)
				throws IOException, InterruptedException {
			HashMapStringIntWritable combined = new HashMapStringIntWritable();
			for (HashMapStringIntWritable value : values) {
				combined.plus(value);
			}

			// Get total count for this word (using empty string as special key)
			int total = 0;
			if (combined.containsKey("")) {
				total = combined.get("");
			}

			// Output total count for this word
			RESULT_KEY.set(key.toString());
			RESULT_VALUE.set(String.valueOf((double) total));
			context.write(RESULT_KEY, RESULT_VALUE);

			// Output relative frequencies for each successor
			for (Map.Entry<String, Integer> entry : combined.entrySet()) {
				String nextWord = entry.getKey();
				// Skip the special total count entry
				if (nextWord.isEmpty()) continue;

				int count = entry.getValue();
				double relativeFreq = (double) count / total;

				RESULT_KEY.set(key.toString() + "\t" + nextWord);
				RESULT_VALUE.set(String.valueOf(relativeFreq));
				context.write(RESULT_KEY, RESULT_VALUE);
			}
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public BigramFrequencyStripes() {
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
		String outputPath = cmdline.getOptionValue(OUTPUT);
		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
				.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

		LOG.info("Tool: " + BigramFrequencyStripes.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Create and configure a MapReduce job
		Configuration conf = getConf();
		Job job = Job.getInstance(conf);
		job.setJobName(BigramFrequencyStripes.class.getSimpleName());
		job.setJarByClass(BigramFrequencyStripes.class);

		job.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(HashMapStringIntWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		/*
		 * A MapReduce program consists of four components: a mapper, a reducer,
		 * an optional combiner, and an optional partitioner.
		 */
		job.setMapperClass(MyMapper.class);
		job.setCombinerClass(MyCombiner.class);
		job.setReducerClass(MyReducer.class);

		// Delete the output directory if it exists already.
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf).delete(outputDir, true);

		// Time the program
		long startTime = System.currentTimeMillis();
		job.waitForCompletion(true);
		LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
				/ 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new BigramFrequencyStripes(), args);
	}
}
