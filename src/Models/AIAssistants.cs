using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.ML.OnnxRuntime;
using Tokenizer = Microsoft.ML.OnnxRuntimeGenAI.Tokenizer;
using System.Threading;

namespace SourceGit.Models
{

    public interface IAIChat
    {
        void Setup();
        Task<AIChatResponse> Chat(string prompt, string question, CancellationToken cancellationToken);

    }

    public class AIChatMessage
    {
        [JsonPropertyName("role")]
        public string Role
        {
            get;
            set;
        }

        [JsonPropertyName("content")]
        public string Content
        {
            get;
            set;
        }
    }
    public class AIChatResponse
    {
        [JsonPropertyName("content")]
        public string Content
        {
            get;
            set;
        }
    }

    public class AIChatRequest
    {
        [JsonPropertyName("model")]
        public string Model
        {
            get;
            set;
        }

        [JsonPropertyName("messages")]
        public List<AIChatMessage> Messages
        {
            get;
            set;
        } = [];

        public void AddMessage(string role, string content)
        {
            Messages.Add(new AIChatMessage { Role = role, Content = content });
        }
    }

    internal class AIAssistant : IAIChat
    {

        private Model model;

        private Tokenizer tokenizer;


        public Task<AIChatResponse> Chat(string prompt, string question, CancellationToken cancellationToken)
        {

            var fullPrompt = $"<|system|>{prompt}<|end|><|user|>{question}<|end|><|assistant|>";

            var sequences = tokenizer.Encode(fullPrompt);

            var generatorParams = new GeneratorParams(model);

            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetSearchOption("past_present_share_buffer", false);
            generatorParams.SetInputSequences(sequences);

            var generator = new Generator(model, generatorParams);

            StringBuilder stringBuilder = new StringBuilder();

            while (!generator.IsDone())
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    return null;
                }

                generator.ComputeLogits();
                generator.GenerateNextToken();
                var outputTokens = generator.GetSequence(0);
                var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
                var output = tokenizer.Decode(newToken);
                stringBuilder.Append(output);
            }
            return Task.FromResult(new AIChatResponse { Content = stringBuilder.ToString() });
        }


        public void Setup()
        {
            string onnxFile = @"..\..\Phi-3.5-mini-instruct-onnx\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4";

            model = new Model(onnxFile);

            tokenizer = new Tokenizer(model);
        }
    }
}
