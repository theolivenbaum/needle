using System.Net.Http;
using System.Security.Cryptography;

namespace Needle.Tokenizer;

/// <summary>
/// Downloads the published <c>needle.model</c> SentencePiece tokenizer
/// from the Cactus-Compute/needle HuggingFace repository and caches it on
/// disk for reuse.  Kept as a maintenance utility for refreshing the
/// embedded <c>Resources/needle.model</c>; the runtime path is
/// <see cref="NeedleTokenizer.LoadDefault"/>, which reads the embedded
/// resource and needs no network.
/// </summary>
internal static class NeedleTokenizerDownloader
{
    private const string HuggingFaceRepo  = "Cactus-Compute/needle";
    private const string TokenizerSubpath = "tokenizer/needle.model";
    private const string FileName         = "needle.model";

    private static readonly object _lock = new();
    private static HttpClient? _http;

    /// <summary>
    /// Default cache directory: <c>~/.cache/needle</c> on Unix /
    /// <c>%LOCALAPPDATA%\needle</c> on Windows.  Can be overridden by
    /// setting the <c>NEEDLE_CACHE_DIR</c> environment variable.
    /// </summary>
    internal static string DefaultCacheDir
    {
        get
        {
            var env = Environment.GetEnvironmentVariable("NEEDLE_CACHE_DIR");
            if (!string.IsNullOrEmpty(env)) return env;

            // ~/.cache/needle on Linux/macOS; %LOCALAPPDATA%\needle on Windows.
            string baseDir = Environment.GetFolderPath(
                Environment.SpecialFolder.LocalApplicationData);
            if (string.IsNullOrEmpty(baseDir))
            {
                baseDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    ".cache");
            }
            return Path.Combine(baseDir, "needle");
        }
    }

    /// <summary>
    /// Ensure the tokenizer <c>.model</c> file is available locally,
    /// downloading from HuggingFace on first call.  Subsequent calls
    /// return immediately from the cache.
    /// </summary>
    /// <param name="cacheDir">
    /// Override the cache directory.  Defaults to <see cref="DefaultCacheDir"/>.
    /// </param>
    /// <returns>Absolute path to the cached <c>needle.model</c> file.</returns>
    /// <exception cref="HttpRequestException">Download failed.</exception>
    internal static string EnsureTokenizerModel(string? cacheDir = null)
    {
        cacheDir ??= DefaultCacheDir;
        string localPath = Path.Combine(cacheDir, FileName);

        // Fast path: already cached and non-empty.
        if (IsValidLocalFile(localPath)) return localPath;

        lock (_lock)
        {
            // Re-check under the lock in case another thread just downloaded.
            if (IsValidLocalFile(localPath)) return localPath;

            Directory.CreateDirectory(cacheDir);
            string url = $"https://huggingface.co/{HuggingFaceRepo}/resolve/main/{TokenizerSubpath}";

            // Download to a temp file then atomically rename, so a partial
            // download from a crashed process can't poison the cache.
            string tempPath = localPath + ".downloading-" + Guid.NewGuid().ToString("N").AsSpan(0, 8).ToString();
            try
            {
                _http ??= new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
                _http.DefaultRequestHeaders.UserAgent.ParseAdd("needle-dotnet/0.1");

                using (var response = _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult())
                {
                    response.EnsureSuccessStatusCode();
                    using var src = response.Content.ReadAsStream();
                    using var dst = File.Create(tempPath);
                    src.CopyTo(dst);
                }

                if (File.Exists(localPath)) File.Delete(localPath);
                File.Move(tempPath, localPath);
            }
            catch
            {
                if (File.Exists(tempPath))
                {
                    try { File.Delete(tempPath); }
                    catch { /* ignore */ }
                }
                throw;
            }

            return localPath;
        }
    }

    /// <summary>
    /// Load the tokenizer, downloading the model file on first call.
    /// Convenience wrapper that combines <see cref="EnsureTokenizerModel"/>
    /// with <see cref="NeedleTokenizer"/> construction.
    /// </summary>
    internal static NeedleTokenizer LoadOrDownload(string? cacheDir = null) =>
        new NeedleTokenizer(EnsureTokenizerModel(cacheDir));

    private static bool IsValidLocalFile(string path)
    {
        if (!File.Exists(path)) return false;
        var info = new FileInfo(path);
        // SentencePiece models are at minimum a few kilobytes; reject
        // obviously truncated cache entries so a re-download can recover.
        return info.Length > 1024;
    }
}
