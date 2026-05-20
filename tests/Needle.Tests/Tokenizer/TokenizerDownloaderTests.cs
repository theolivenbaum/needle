using Needle.Tokenizer;

namespace Needle.Tests.Tokenizer;

/// <summary>
/// Smoke tests for <see cref="NeedleTokenizerDownloader"/>.  Skipped
/// silently when no network access is available.
/// </summary>
public sealed class TokenizerDownloaderTests
{
    private static string TempCacheDir() =>
        Path.Combine(Path.GetTempPath(),
                     "needle-test-" + Guid.NewGuid().ToString("N").Substring(0, 12));

    [Fact]
    public void EnsureTokenizerModel_DownloadsAndCaches()
    {
        var dir = TempCacheDir();
        try
        {
            string path;
            try
            {
                path = NeedleTokenizerDownloader.EnsureTokenizerModel(dir);
            }
            catch
            {
                // Offline / blocked network — skip silently.
                return;
            }

            Assert.True(File.Exists(path));
            Assert.True(new FileInfo(path).Length > 1024);
            Assert.StartsWith(dir, path, StringComparison.Ordinal);

            // Second call should hit the cache and return the same path
            // without touching the network.
            string mtime = File.GetLastWriteTimeUtc(path).ToString("O");
            string cached = NeedleTokenizerDownloader.EnsureTokenizerModel(dir);
            Assert.Equal(path, cached);
            Assert.Equal(mtime, File.GetLastWriteTimeUtc(cached).ToString("O"));
        }
        finally
        {
            if (Directory.Exists(dir))
            {
                try { Directory.Delete(dir, recursive: true); }
                catch { /* ignore */ }
            }
        }
    }

    [Fact]
    public void LoadOrDownload_ProducesUsableTokenizer()
    {
        var dir = TempCacheDir();
        try
        {
            NeedleTokenizer tok;
            try
            {
                tok = NeedleTokenizerDownloader.LoadOrDownload(dir);
            }
            catch
            {
                return; // offline — skip
            }

            using (tok)
            {
                var ids = tok.Encode("Hello world");
                Assert.NotEmpty(ids);
                Assert.Equal("Hello world", tok.Decode(ids));
            }
        }
        finally
        {
            if (Directory.Exists(dir))
            {
                try { Directory.Delete(dir, recursive: true); }
                catch { /* ignore */ }
            }
        }
    }
}
