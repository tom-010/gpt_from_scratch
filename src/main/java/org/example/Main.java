package org.example;
import java.io.*;
import java.nio.file.*;
import java.util.*;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.*;
import com.google.gson.JsonObject;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

class Encoder {

    Map<Integer, Integer> codePointToEncoding = new HashMap<>();
    Map<Integer, Integer> encodingToCodePoint = new HashMap<>();

    List<Integer> alphabet;

    void init(String text) {
        var codePoints = Arrays.stream(text.chars().toArray()).boxed().toList();
        var unique = new HashSet<>(codePoints);
        alphabet = unique.stream().sorted().toList();

        for(int i=0; i<alphabet.size(); i++) {
            codePointToEncoding.put(alphabet.get(i), i);
            encodingToCodePoint.put(i, alphabet.get(i));
        }
    }

    List<Integer> encode(String s) {
        return s.codePoints().map(c -> codePointToEncoding.get(c)).boxed().toList();
    }

    String decode(List<Integer> encoded) {
        return encoded.stream()
                .map(e -> encodingToCodePoint.get(e))
                .filter(Objects::nonNull)
                .collect( StringBuilder::new , StringBuilder::appendCodePoint , StringBuilder::append)
                .toString();
    }

    String getAlphabetString() {
        return alphabet.stream()
                .filter(Objects::nonNull)
                .collect( StringBuilder::new , StringBuilder::appendCodePoint , StringBuilder::append)
                .toString();
    }

    String decode(Integer integer) {
        return decode(List.of(integer));
    }

}

class Sample {
    List<Integer> input;
    List<Integer> expected;

    Sample(List<Integer> input, List<Integer> expected) {
        this.input = input;
        this.expected = expected;
    }

    int size() {
        return input.size();
    }

    @Override
    public String toString() {
        return "Sample{" +
                "input=" + input +
                ", expected=" + expected +
                '}';
    }
}

class DataSource {

    Encoder encoder;
    String path;

    List<Integer> encoded;

    List<Integer> trainingData;

    List<Integer> testData;

    Random ran = new Random();



    DataSource(String path) {
        this.path = path;
    }

    DataSource() {
        this.path = null;
    }

    void read() throws IOException {
        String content = readFile();
        encoder = new Encoder();
        encoder.init(content);
        encoded = encoder.encode(content);
        splitData();
    }

    private String readFile() throws IOException {
        File file = new File(path);
        InputStream inputStream = new FileInputStream(file);


        StringBuilder resultStringBuilder = new StringBuilder();
        try (BufferedReader br
                     = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = br.readLine()) != null) {
                resultStringBuilder.append(line).append("\n");
            }
        }
        return resultStringBuilder.toString();
    }

    void splitData() {
        int i = (int) (0.9 * encoded.size());
        trainingData = encoded.subList(0, i);
        testData = encoded.subList(i, encoded.size());
    }

    Sample getTrainingSample(int contextLength) {
        return drawFromList(trainingData, contextLength);
    }

    Sample getTestSample(int contextLength) {
        return drawFromList(trainingData, contextLength);
    }

    private Sample drawFromList(List<Integer> list, int contextLength) {
        int max = list.size() - (contextLength+1);
        int position = ran.nextInt(max);

        return new Sample(
                list.subList(position, position + contextLength),
                list.subList(position+1, position + contextLength + 1));
    }

    void store(String filename) throws IOException {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("trainingData", trainingData);
        jsonObject.put("testData", testData);
        jsonObject.put("codePointToEncoding", encoder.codePointToEncoding);
        jsonObject.put("encodingToCodePoint", encoder.encodingToCodePoint);
        FileWriter file = new FileWriter(filename);
        jsonObject.writeJSONString(file);
        file.close();
    }

    static DataSource fromCache(String filename) throws IOException, ParseException {
        JSONParser parser = new JSONParser();
        FileReader file = new FileReader(filename);
        var parsed = (JsonObject) parser.parse(file);
        var trainingData = parsed.get("trainingData");
        System.out.println(trainingData);
        return new DataSource();
    }
}

class GptTranslator implements Translator<List<Integer>, Integer> {

    @Override
    public NDList processInput(TranslatorContext ctx, List<Integer> input) {
        NDManager manager = ctx.getNDManager();
        int[] primitive = input.stream().mapToInt(Integer::intValue).toArray();
        NDArray array = manager.create(primitive);
        return new NDList (array);
    }

    @Override
    public Integer processOutput(TranslatorContext ctx, NDList logits) {
        var probabilities = logits.get(0).softmax(-1).toFloatArray();

        return ctx.getNDManager().randomMultinomial(1, logits.get(0).softmax(-1)).getInt();

        /*var m = new Multinomial<Integer>();
        for(int i=0; i<probabilities.length; i++) {
            m.add(i, probabilities[i]);
        }
        return m.sample();*/
    }
}

class BashTranslator {

}

public class Main {

    public static void main(String[] args) throws MalformedModelException, IOException, ParseException, TranslateException {
        Path modelDir = Paths.get("./");
        Model model = Model.newInstance("gpt_gpu");
        model.load(modelDir);
        Predictor<List<Integer>, Integer> predictor = model.newPredictor(new GptTranslator());
        predictor.predict(List.of(0));
    }


    public static void main2(String[] args) throws IOException, MalformedModelException, TranslateException {
        DataSource ds = new DataSource("input.txt");
        ds.read();
        var encoder = ds.encoder;
        System.out.println(encoder.getAlphabetString());

        Path modelDir = Paths.get("./");
        Model model = Model.newInstance("gpt");
        model.load(modelDir);

        Predictor<List<Integer>, Integer> predictor = model.newPredictor(new GptTranslator());

        int contextSize = 64;
        List<Integer> context = new LinkedList<Integer>();
        context.add(0); // initial token

        for(int i=0; i<500; i++) {
            int nextToken = predictor.predict(List.of(50));
            context.add(nextToken);
            while(context.size() > contextSize) {
                context.remove(0);
            }
            System.out.print(encoder.decode(nextToken));
        }
        System.out.println();
    }


}
