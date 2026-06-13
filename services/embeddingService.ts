/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { AutoTokenizer, AutoModel, env, matmul } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';
import type { RankedDocument } from '../types';

env.allowLocalModels = false;
env.allowRemoteModels = true;

// You can try 'https://hf-mirror.com' if 'https://huggingface.co' (default) is unstable in your region
// env.remoteHost = 'https://hf-mirror.com';

const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const DTYPE = "q4"; // Options: "fp32" | "q8" | "q4"

const PREFIXES = {
  query: "task: search result | query: ",
  document: "title: none | text: ",
};

// Singleton instance
let instance: EmbeddingService | null = null;

class EmbeddingService {
  private tokenizer: AutoTokenizer | null = null;
  private model: AutoModel | null = null;
  private ready: boolean = false;

  private constructor(progressCallback: (progress: any) => void) {
    this.init(progressCallback);
  }

  static async getInstance(progressCallback: (progress: any) => void): Promise<EmbeddingService> {
    if (instance === null) {
      instance = new EmbeddingService(progressCallback);
      await instance.init(progressCallback);
    }
    return instance;
  }

  private async init(progressCallback: (progress: any) => void): Promise<void> {
    if (this.ready) {
      return;
    }

    try {
      this.tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: progressCallback });
      this.model = await AutoModel.from_pretrained(MODEL_ID, {
        dtype: DTYPE,
        progress_callback: progressCallback,
      });
      this.ready = true;
    } catch (error) {
      console.error("Failed to initialize embedding service:", error);
      throw error;
    }
  }

  async embed(query: string, documents: string[]): Promise<{ rankedDocuments: RankedDocument[], queryEmbedding: number[], documentEmbeddings: number[][] }> {
    if (!this.tokenizer || !this.model || !this.ready) {
      throw new Error("Embedding service is not ready.");
    }

    const prefixedQuery = PREFIXES.query + query;
    const prefixedDocs = documents.map(doc => PREFIXES.document + doc);

    const inputs = await this.tokenizer([prefixedQuery, ...prefixedDocs], { padding: true, truncation: true });

    const { sentence_embedding } = await this.model(inputs);

    const embeddings = sentence_embedding.tolist() as number[][];
    const queryEmbedding = embeddings[0];
    const documentEmbeddings = embeddings.slice(1);

    if (documents.length === 0) {
      return { rankedDocuments: [], queryEmbedding, documentEmbeddings };
    }

    const scores = await matmul(sentence_embedding, sentence_embedding.transpose(1, 0));

    const similarities = (scores.tolist() as number[][])[0].slice(1);

    const ranking: RankedDocument[] = similarities
      .map((score, index) => ({
        index: index,
        text: documents[index],
        score: score,
      }))
      .sort((a, b) => b.score - a.score);

    return { rankedDocuments: ranking, queryEmbedding, documentEmbeddings };
  }
}

export default EmbeddingService;
