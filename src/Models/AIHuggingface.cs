using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SourceGit.Models
{


    internal class AIHuggingfaceChatResponse
    {

        //List<AIHuggingfaceChatMessage> Messages { get; set; }
        string Content;

    }

    internal record AIHuggingfaceChatMessage(string generated_text);

    internal class AIHuggingface : IAIChat
    {
        private string ModelName { get; set; }
        private string Token { get; set; }
        public AIHuggingface()
        {
            Token = "";
            ModelName = "mistralai/Mistral-7B-Instruct-v0.3";
        }

        public void Setup()
        {
            // Perform any setup logic here
        }

        public async Task<AIChatResponse> Chat(string prompt, string question, CancellationToken cancellationToken)
        {
            var httpClient = new HttpClient();
            // Create the HTTP request
            var request = new HttpRequestMessage(HttpMethod.Post, $"https://api-inference.huggingface.co/models/{ModelName}");
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", Token);
            request.Content = JsonContent.Create(new
            {
                inputs = $"{prompt}\n{question}"

            });

            // Send the HTTP request and get the response
            var response = await httpClient.SendAsync(request, cancellationToken);


            if (!response.IsSuccessStatusCode)
            {
                throw new HttpRequestException($"API call failure: {response.StatusCode}");
            }

            // Read the response content
            var responseContent = await response.Content.ReadAsStringAsync();




            // Deserialize the response content


            var chatResponse = System.Text.Json.JsonSerializer.Deserialize<List<AIHuggingfaceChatMessage>>(responseContent);

            return new AIChatResponse
            {
                Content = chatResponse[0].generated_text
            };
        }
    }

}
