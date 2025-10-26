'use client';

import { useState } from 'react';

interface TranscriptionResult {
  text: string;
  prediction: string;
  probabilities: {
    yes: number;
    no: number;
    maybe_confusion: number;
  };
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setError('Please select an audio file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('audio', file);

      const response = await fetch('http://localhost:8000/transcribe-and-classify', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getPredictionColor = (prediction: string) => {
    switch (prediction.toLowerCase()) {
      case 'yes':
        return 'text-green-600 dark:text-green-400';
      case 'no':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-yellow-600 dark:text-yellow-400';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-indigo-950 dark:to-purple-950 animate-gradient relative overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-400/20 dark:bg-indigo-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-400/20 dark:bg-purple-500/10 rounded-full blur-3xl"></div>
      </div>

      <main className="container mx-auto px-4 py-16 max-w-4xl relative z-10">
        <div className="text-center mb-12 animate-fade-in-up">
          <div className="inline-block mb-4">
            <div className="flex items-center justify-center w-20 h-20 mx-auto bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl shadow-lg mb-4">
              <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
          </div>
          <h1 className="text-6xl font-extrabold bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-4 tracking-tight">
            Audio Intelligence
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto font-medium">
            Advanced AI-powered transcription and classification for your audio files
          </p>
        </div>

        {/* Baking & Cooking Section */}
        <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 dark:border-slate-700/50 p-8 mb-8 animate-scale-in">
          <h2 className="text-3xl font-bold text-center bg-gradient-to-r from-orange-600 to-red-600 dark:from-orange-400 dark:to-red-400 bg-clip-text text-transparent mb-8">
            Baking & Cooking Studio
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-2xl p-6 border border-orange-200 dark:border-orange-800/50">
              <div className="text-center mb-4">
                <div className="text-6xl mb-4">🍪</div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2">Cookie Recipes</h3>
                <p className="text-gray-600 dark:text-gray-400">Delicious homemade cookies for every occasion</p>
              </div>
              <img
                src="/cookies.jpg"
                alt="Fresh baked cookies"
                className="w-full h-48 object-cover rounded-xl"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 rounded-2xl p-6 border border-amber-200 dark:border-amber-800/50">
              <div className="text-center mb-4">
                <div className="text-6xl mb-4">🔥</div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2">Kitchen Oven</h3>
                <p className="text-gray-600 dark:text-gray-400">Professional baking equipment</p>
              </div>
              <img
                src="/oven.jpg"
                alt="Modern kitchen oven"
                className="w-full h-48 object-cover rounded-xl"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>
          </div>
        </div>

        {/* Notable Figures Gallery */}
        <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 dark:border-slate-700/50 p-8 mb-8 animate-scale-in">
          <h2 className="text-3xl font-bold text-center bg-gradient-to-r from-purple-600 to-indigo-600 dark:from-purple-400 dark:to-indigo-400 bg-clip-text text-transparent mb-8">
            Notable Figures in History
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="group relative overflow-hidden rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 bg-gradient-to-br from-purple-100 to-indigo-100 dark:from-purple-900/30 dark:to-indigo-900/30 p-6">
              <div className="text-center">
                <div className="text-6xl mb-4">🎭</div>
                <h3 className="text-lg font-bold text-gray-800 dark:text-gray-200">Cultural Icons</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">Artists and performers</p>
              </div>
            </div>
            <div className="group relative overflow-hidden rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 bg-gradient-to-br from-blue-100 to-cyan-100 dark:from-blue-900/30 dark:to-cyan-900/30 p-6">
              <div className="text-center">
                <div className="text-6xl mb-4">🔬</div>
                <h3 className="text-lg font-bold text-gray-800 dark:text-gray-200">Scientists</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">Innovators and researchers</p>
              </div>
            </div>
            <div className="group relative overflow-hidden rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 bg-gradient-to-br from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 p-6">
              <div className="text-center">
                <div className="text-6xl mb-4">📚</div>
                <h3 className="text-lg font-bold text-gray-800 dark:text-gray-200">Authors</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">Writers and thinkers</p>
              </div>
            </div>
          </div>
        </div>

        {/* World Leaders Gallery */}
        <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 dark:border-slate-700/50 p-8 mb-8 animate-scale-in">
          <h2 className="text-3xl font-bold text-center bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-300 bg-clip-text text-transparent mb-8">
            World Leaders
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Zelensky */}
            <div className="group relative overflow-hidden rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105">
              <div className="aspect-square bg-gradient-to-br from-blue-100 to-yellow-100 dark:from-blue-900/30 dark:to-yellow-900/30 flex items-center justify-center">
                <img
                  src="/zelensky.jpg"
                  alt="Volodymyr Zelensky"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                    e.currentTarget.parentElement!.innerHTML += '<div class="text-center p-8"><div class="text-6xl mb-4">🇺🇦</div><p class="text-lg font-bold text-gray-700 dark:text-gray-300">Volodymyr Zelensky</p><p class="text-sm text-gray-500 dark:text-gray-400 mt-2">President of Ukraine</p></div>';
                  }}
                />
              </div>
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                <h3 className="text-white font-bold text-lg">Volodymyr Zelensky</h3>
                <p className="text-gray-200 text-sm">President of Ukraine</p>
              </div>
            </div>

            {/* Putin */}
            <div className="group relative overflow-hidden rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105">
              <div className="aspect-square bg-gradient-to-br from-red-100 to-blue-100 dark:from-red-900/30 dark:to-blue-900/30 flex items-center justify-center">
                <img
                  src="/putin.jpg"
                  alt="Vladimir Putin"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                    e.currentTarget.parentElement!.innerHTML += '<div class="text-center p-8"><div class="text-6xl mb-4">🇷🇺</div><p class="text-lg font-bold text-gray-700 dark:text-gray-300">Vladimir Putin</p><p class="text-sm text-gray-500 dark:text-gray-400 mt-2">President of Russia</p></div>';
                  }}
                />
              </div>
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                <h3 className="text-white font-bold text-lg">Vladimir Putin</h3>
                <p className="text-gray-200 text-sm">President of Russia</p>
              </div>
            </div>

            {/* Trump */}
            <div className="group relative overflow-hidden rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105">
              <div className="aspect-square bg-gradient-to-br from-red-100 to-blue-100 dark:from-red-900/30 dark:to-blue-900/30 flex items-center justify-center">
                <img
                  src="/trump.jpg"
                  alt="Donald Trump"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                    e.currentTarget.parentElement!.innerHTML += '<div class="text-center p-8"><div class="text-6xl mb-4">🇺🇸</div><p class="text-lg font-bold text-gray-700 dark:text-gray-300">Donald Trump</p><p class="text-sm text-gray-500 dark:text-gray-400 mt-2">47th President of the USA</p></div>';
                  }}
                />
              </div>
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                <h3 className="text-white font-bold text-lg">Donald Trump</h3>
                <p className="text-gray-200 text-sm">47th President of the USA</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 dark:border-slate-700/50 p-8 mb-8 animate-scale-in">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label
                htmlFor="audio-upload"
                className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3"
              >
                Select Audio File
              </label>
              <div className="flex items-center justify-center w-full">
                <label
                  htmlFor="audio-upload"
                  className="flex flex-col items-center justify-center w-full h-48 border-2 border-indigo-300 dark:border-indigo-700/50 border-dashed rounded-2xl cursor-pointer bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950/50 hover:from-indigo-100 hover:to-purple-100 dark:hover:from-slate-800 dark:hover:to-indigo-900/50 transition-all duration-300 hover:shadow-lg hover:scale-[1.02] group"
                >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <div className="mb-4 p-4 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl group-hover:scale-110 transition-transform duration-300">
                      <svg
                        className="w-8 h-8 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        />
                      </svg>
                    </div>
                    <p className="mb-2 text-base text-gray-700 dark:text-gray-300">
                      <span className="font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Click to upload</span>
                      <span className="font-medium"> or drag and drop</span>
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      MP3, WAV, FLAC (MAX. 50MB)
                    </p>
                    {file && (
                      <div className="mt-4 px-4 py-2 bg-indigo-100 dark:bg-indigo-900/50 rounded-lg border border-indigo-300 dark:border-indigo-700">
                        <p className="text-sm font-semibold text-indigo-700 dark:text-indigo-300 flex items-center gap-2">
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                          {file.name}
                        </p>
                      </div>
                    )}
                  </div>
                  <input
                    id="audio-upload"
                    type="file"
                    className="hidden"
                    accept="audio/*"
                    onChange={handleFileChange}
                  />
                </label>
              </div>
            </div>

            <button
              type="submit"
              disabled={!file || loading}
              className="w-full bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 disabled:from-gray-400 disabled:via-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed text-white font-bold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl hover:scale-[1.02] disabled:hover:scale-100 group"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-6 w-6" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  <span className="text-lg">Processing...</span>
                </>
              ) : (
                <>
                  <span className="text-lg">Transcribe & Classify</span>
                  <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </>
              )}
            </button>
          </form>

          {error && (
            <div className="mt-6 p-4 bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 rounded-xl shadow-md animate-scale-in">
              <div className="flex items-start gap-3">
                <svg className="w-6 h-6 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="font-bold text-red-700 dark:text-red-400">Error</p>
                  <p className="text-red-600 dark:text-red-300 mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {result && (
          <div className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 dark:border-slate-700/50 p-8 space-y-8 animate-scale-in">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-300 bg-clip-text text-transparent">
                Results
              </h2>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-5 h-5 text-indigo-600 dark:text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200">
                  Transcription
                </h3>
              </div>
              <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950/50 rounded-2xl p-6 border border-indigo-200 dark:border-indigo-800/50 shadow-sm">
                <p className="text-gray-800 dark:text-gray-200 whitespace-pre-wrap leading-relaxed font-medium">
                  {result.text}
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-5 h-5 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200">
                  Classification
                </h3>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-slate-900 dark:to-purple-950/50 rounded-2xl p-6 border border-purple-200 dark:border-purple-800/50 shadow-sm">
                <p className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-4 uppercase tracking-wide">
                  Prediction
                </p>
                <div className="mb-6 p-4 bg-white/60 dark:bg-slate-800/60 backdrop-blur rounded-xl border border-gray-200 dark:border-gray-700">
                  <p className={`text-4xl font-extrabold uppercase tracking-tight ${getPredictionColor(result.prediction)}`}>
                    {result.prediction}
                  </p>
                </div>

                <p className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-3 uppercase tracking-wide">
                  Confidence Scores
                </p>
                <div className="space-y-4">
                  <div className="group">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-base font-bold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                        <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                        Yes
                      </span>
                      <span className="text-base font-bold text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30 px-3 py-1 rounded-lg">
                        {(result.probabilities.yes * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden shadow-inner">
                      <div
                        className="bg-gradient-to-r from-green-500 to-emerald-500 h-3 rounded-full transition-all duration-700 ease-out shadow-lg"
                        style={{ width: `${result.probabilities.yes * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="group">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-base font-bold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                        <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                        No
                      </span>
                      <span className="text-base font-bold text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30 px-3 py-1 rounded-lg">
                        {(result.probabilities.no * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden shadow-inner">
                      <div
                        className="bg-gradient-to-r from-red-500 to-rose-500 h-3 rounded-full transition-all duration-700 ease-out shadow-lg"
                        style={{ width: `${result.probabilities.no * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="group">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-base font-bold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                        <span className="w-2 h-2 bg-yellow-500 rounded-full"></span>
                        Maybe / Confusion
                      </span>
                      <span className="text-base font-bold text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30 px-3 py-1 rounded-lg">
                        {(result.probabilities.maybe_confusion * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden shadow-inner">
                      <div
                        className="bg-gradient-to-r from-yellow-500 to-amber-500 h-3 rounded-full transition-all duration-700 ease-out shadow-lg"
                        style={{ width: `${result.probabilities.maybe_confusion * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
