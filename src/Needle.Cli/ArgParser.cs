namespace Needle.Cli;

/// <summary>
/// Tiny argument parser that handles <c>--key value</c> and <c>--flag</c>
/// style arguments.  No dependency on System.CommandLine to keep the
/// binary lean.
/// </summary>
internal sealed class ArgParser
{
    private readonly Dictionary<string, string> _values;
    private readonly HashSet<string>            _flags;

    private ArgParser(Dictionary<string, string> values, HashSet<string> flags)
    {
        _values = values;
        _flags  = flags;
    }

    public static ArgParser Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        var flags  = new HashSet<string>(StringComparer.Ordinal);

        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (!a.StartsWith("--", StringComparison.Ordinal)) continue;
            string key = a[2..];

            if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                values[key] = args[++i];
            }
            else
            {
                flags.Add(key);
            }
        }

        return new ArgParser(values, flags);
    }

    public string? GetOrNull(string key) =>
        _values.TryGetValue(key, out var v) ? v : null;

    public string Get(string key, string fallback) =>
        _values.TryGetValue(key, out var v) ? v : fallback;

    public string GetRequired(string key)
    {
        if (_values.TryGetValue(key, out var v)) return v;
        throw new InvalidOperationException($"Missing required argument: --{key}");
    }

    public bool GetFlag(string key) => _flags.Contains(key);
}
