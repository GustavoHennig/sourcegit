using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

using Avalonia;
using Avalonia.Media.Imaging;

namespace SourceGit.Models
{
    public enum TextDiffLineType
    {
        None,
        Normal,
        Indicator,
        Added,
        Deleted,
    }

    public class TextInlineRange(int p, int n)
    {
        public int Start { get; set; } = p;
        public int End { get; set; } = p + n - 1;
    }

    public class TextDiffLine
    {
        public TextDiffLineType Type { get; set; } = TextDiffLineType.None;
        public string Content { get; set; } = "";
        public int OldLineNumber { get; set; } = 0;
        public int NewLineNumber { get; set; } = 0;
        public List<TextInlineRange> Highlights { get; set; } = new List<TextInlineRange>();
        public bool NoNewLineEndOfFile { get; set; } = false;

        public string OldLine => OldLineNumber == 0 ? string.Empty : OldLineNumber.ToString();
        public string NewLine => NewLineNumber == 0 ? string.Empty : NewLineNumber.ToString();

        public TextDiffLine() { }
        public TextDiffLine(TextDiffLineType type, string content, int oldLine, int newLine)
        {
            Type = type;
            Content = content;
            OldLineNumber = oldLine;
            NewLineNumber = newLine;
        }
    }

    public class TextDiffSelection
    {
        public int StartLine { get; set; } = 0;
        public int EndLine { get; set; } = 0;
        public bool HasChanges { get; set; } = false;
        public bool HasLeftChanges { get; set; } = false;
        public int IgnoredAdds { get; set; } = 0;
        public int IgnoredDeletes { get; set; } = 0;

        public bool IsInRange(int idx)
        {
            return idx >= StartLine - 1 && idx < EndLine;
        }
    }

    public partial class TextDiff
    {
        public string File { get; set; } = string.Empty;
        public List<TextDiffLine> Lines { get; set; } = new List<TextDiffLine>();
        public Vector ScrollOffset { get; set; } = Vector.Zero;
        public int MaxLineNumber = 0;

        public string Repo { get; set; } = null;
        public DiffOption Option { get; set; } = null;

        public TextDiffSelection MakeSelection(int startLine, int endLine, bool isCombined, bool isOldSide)
        {
            var rs = new TextDiffSelection();
            rs.StartLine = startLine;
            rs.EndLine = endLine;

            for (int i = 0; i < startLine - 1; i++)
            {
                var line = Lines[i];
                if (line.Type == TextDiffLineType.Added)
                {
                    rs.HasLeftChanges = true;
                    rs.IgnoredAdds++;
                }
                else if (line.Type == TextDiffLineType.Deleted)
                {
                    rs.HasLeftChanges = true;
                    rs.IgnoredDeletes++;
                }
            }

            for (int i = startLine - 1; i < endLine; i++)
            {
                var line = Lines[i];
                if (line.Type == TextDiffLineType.Added)
                {
                    if (isCombined)
                    {
                        rs.HasChanges = true;
                        break;
                    }
                    if (isOldSide)
                    {
                        rs.HasLeftChanges = true;
                    }
                    else
                    {
                        rs.HasChanges = true;
                    }
                }
                else if (line.Type == TextDiffLineType.Deleted)
                {
                    if (isCombined)
                    {
                        rs.HasChanges = true;
                        break;
                    }
                    if (isOldSide)
                    {
                        rs.HasChanges = true;
                    }
                    else
                    {
                        rs.HasLeftChanges = true;
                    }
                }
            }

            if (!rs.HasLeftChanges)
            {
                for (int i = endLine; i < Lines.Count; i++)
                {
                    var line = Lines[i];
                    if (line.Type == TextDiffLineType.Added || line.Type == TextDiffLineType.Deleted)
                    {
                        rs.HasLeftChanges = true;
                        break;
                    }
                }
            }

            return rs;
        }

        public void GenerateNewPatchFromSelection(Change change, string fileBlobGuid, TextDiffSelection selection, bool revert, string output)
        {
            var isTracked = !string.IsNullOrEmpty(fileBlobGuid);
            var fileGuid = isTracked ? fileBlobGuid : "00000000";

            using var writer = new StreamWriter(output);
            writer.WriteLine($"diff --git a/{change.Path} b/{change.Path}");
            if (!revert && !isTracked)
                writer.WriteLine("new file mode 100644");
            writer.WriteLine($"index 00000000...{fileGuid}");
            writer.WriteLine($"--- {(revert || isTracked ? $"a/{change.Path}" : "/dev/null")}");
            writer.WriteLine($"+++ b/{change.Path}");

            var additions = selection.EndLine - selection.StartLine;
            if (selection.StartLine != 1)
                additions++;

            if (revert)
            {
                var totalLines = Lines.Count - 1;
                writer.WriteLine($"@@ -0,{totalLines - additions} +0,{totalLines} @@");
                for (int i = 1; i <= totalLines; i++)
                {
                    var line = Lines[i];
                    if (line.Type != TextDiffLineType.Added)
                        continue;
                    writer.WriteLine($"{(selection.IsInRange(i) ? "+" : " ")}{line.Content}");
                }
            }
            else
            {
                writer.WriteLine($"@@ -0,0 +0,{additions} @@");
                for (int i = selection.StartLine - 1; i < selection.EndLine; i++)
                {
                    var line = Lines[i];
                    if (line.Type != TextDiffLineType.Added)
                        continue;
                    writer.WriteLine($"+{line.Content}");
                }
            }

            writer.WriteLine("\\ No newline at end of file");
            writer.Flush();
        }

        public void GeneratePatchFromSelection(Change change, string fileTreeGuid, TextDiffSelection selection, bool revert, string output)
        {
            var orgFile = !string.IsNullOrEmpty(change.OriginalPath) ? change.OriginalPath : change.Path;

            using var writer = new StreamWriter(output);
            writer.WriteLine($"diff --git a/{change.Path} b/{change.Path}");
            writer.WriteLine($"index 00000000...{fileTreeGuid} 100644");
            writer.WriteLine($"--- a/{orgFile}");
            writer.WriteLine($"+++ b/{change.Path}");

            // If last line of selection is a change. Find one more line.
            string tail = null;
            if (selection.EndLine < Lines.Count)
            {
                var lastLine = Lines[selection.EndLine - 1];
                if (lastLine.Type == TextDiffLineType.Added || lastLine.Type == TextDiffLineType.Deleted)
                {
                    for (int i = selection.EndLine; i < Lines.Count; i++)
                    {
                        var line = Lines[i];
                        if (line.Type == TextDiffLineType.Indicator)
                            break;
                        if (line.Type == TextDiffLineType.Normal ||
                            (revert && line.Type == TextDiffLineType.Added) ||
                            (!revert && line.Type == TextDiffLineType.Deleted))
                        {
                            tail = line.Content;
                            break;
                        }
                    }
                }
            }

            // If the first line is not indicator.
            if (Lines[selection.StartLine - 1].Type != TextDiffLineType.Indicator)
            {
                var indicator = selection.StartLine - 1;
                for (int i = selection.StartLine - 2; i >= 0; i--)
                {
                    var line = Lines[i];
                    if (line.Type == TextDiffLineType.Indicator)
                    {
                        indicator = i;
                        break;
                    }
                }

                var ignoreAdds = 0;
                var ignoreRemoves = 0;
                for (int i = 0; i < indicator; i++)
                {
                    var line = Lines[i];
                    if (line.Type == TextDiffLineType.Added)
                    {
                        ignoreAdds++;
                    }
                    else if (line.Type == TextDiffLineType.Deleted)
                    {
                        ignoreRemoves++;
                    }
                }

                for (int i = indicator; i < selection.StartLine - 1; i++)
                {
                    var line = Lines[i];
                    if (line.Type == TextDiffLineType.Indicator)
                    {
                        ProcessIndicatorForPatch(writer, line, i, selection.StartLine, selection.EndLine, ignoreRemoves, ignoreAdds, revert, tail != null);
                    }
                    else if (line.Type == TextDiffLineType.Added)
                    {
                        if (revert)
                            writer.WriteLine($" {line.Content}");
                    }
                    else if (line.Type == TextDiffLineType.Deleted)
                    {
                        if (!revert)
                            writer.WriteLine($" {line.Content}");
                    }
                    else if (line.Type == TextDiffLineType.Normal)
                    {
                        writer.WriteLine($" {line.Content}");
                    }
                }
            }

            // Outputs the selected lines.
            for (int i = selection.StartLine - 1; i < selection.EndLine; i++)
            {
                var line = Lines[i];
                if (line.Type == TextDiffLineType.Indicator)
                {
                    if (!ProcessIndicatorForPatch(writer, line, i, selection.StartLine, selection.EndLine, selection.IgnoredDeletes, selection.IgnoredAdds, revert, tail != null))
                        break;
                }
                else if (line.Type == TextDiffLineType.Normal)
                {
                    writer.WriteLine($" {line.Content}");
                }
                else if (line.Type == TextDiffLineType.Added)
                {
                    writer.WriteLine($"+{line.Content}");
                }
                else if (line.Type == TextDiffLineType.Deleted)
                {
                    writer.WriteLine($"-{line.Content}");
                }
            }

            writer.WriteLine($" {tail}");
            writer.Flush();
        }

        public void GeneratePatchFromSelectionSingleSide(Change change, string fileTreeGuid, TextDiffSelection selection, bool revert, bool isOldSide, string output)
        {
            var orgFile = !string.IsNullOrEmpty(change.OriginalPath) ? change.OriginalPath : change.Path;

            using var writer = new StreamWriter(output);
            writer.WriteLine($"diff --git a/{change.Path} b/{change.Path}");
            writer.WriteLine($"index 00000000...{fileTreeGuid} 100644");
            writer.WriteLine($"--- a/{orgFile}");
            writer.WriteLine($"+++ b/{change.Path}");

            // If last line of selection is a change. Find one more line.
            string tail = null;
            if (selection.EndLine < Lines.Count)
            {
                var lastLine = Lines[selection.EndLine - 1];
                if (lastLine.Type == TextDiffLineType.Added || lastLine.Type == TextDiffLineType.Deleted)
                {
                    for (int i = selection.EndLine; i < Lines.Count; i++)
                    {
                        var line = Lines[i];
                        if (line.Type == TextDiffLineType.Indicator)
                            break;
                        if (revert)
                        {
                            if (line.Type == TextDiffLineType.Normal || line.Type == TextDiffLineType.Added)
                            {
                                tail = line.Content;
                                break;
                            }
                        }
                        else
                        {
                            if (line.Type == TextDiffLineType.Normal || line.Type == TextDiffLineType.Deleted)
                            {
                                tail = line.Content;
                                break;
                            }
                        }
                    }
                }
            }

            // If the first line is not indicator.
            if (Lines[selection.StartLine - 1].Type != TextDiffLineType.Indicator)
            {
                var indicator = selection.StartLine - 1;
                for (int i = selection.StartLine - 2; i >= 0; i--)
                {
                    var line = Lines[i];
                    if (line.Type == TextDiffLineType.Indicator)
                    {
                        indicator = i;
                        break;
                    }
                }

                var ignoreAdds = 0;
                var ignoreRemoves = 0;
                for (int i = 0; i < indicator; i++)
                {
                    var line = Lines[i];
                    if (line.Type == TextDiffLineType.Added)
                    {
                        ignoreAdds++;
                    }
                    else if (line.Type == TextDiffLineType.Deleted)
                    {
                        ignoreRemoves++;
                    }
                }

                for (int i = indicator; i < selection.StartLine - 1; i++)
                {
                    var line = Lines[i];
                    if (line.Type == TextDiffLineType.Indicator)
                    {
                        ProcessIndicatorForPatchSingleSide(writer, line, i, selection.StartLine, selection.EndLine, ignoreRemoves, ignoreAdds, revert, isOldSide, tail != null);
                    }
                    else if (line.Type == TextDiffLineType.Added)
                    {
                        if (revert)
                            writer.WriteLine($" {line.Content}");
                    }
                    else if (line.Type == TextDiffLineType.Deleted)
                    {
                        if (!revert)
                            writer.WriteLine($" {line.Content}");
                    }
                    else if (line.Type == TextDiffLineType.Normal)
                    {
                        writer.WriteLine($" {line.Content}");
                    }
                }
            }

            // Outputs the selected lines.
            for (int i = selection.StartLine - 1; i < selection.EndLine; i++)
            {
                var line = Lines[i];
                if (line.Type == TextDiffLineType.Indicator)
                {
                    if (!ProcessIndicatorForPatchSingleSide(writer, line, i, selection.StartLine, selection.EndLine, selection.IgnoredDeletes, selection.IgnoredAdds, revert, isOldSide, tail != null))
                        break;
                }
                else if (line.Type == TextDiffLineType.Normal)
                {
                    writer.WriteLine($" {line.Content}");
                }
                else if (line.Type == TextDiffLineType.Added)
                {
                    if (isOldSide)
                    {
                        if (revert)
                        {
                            writer.WriteLine($" {line.Content}");
                        }
                        else
                        {
                            selection.IgnoredAdds++;
                        }
                    }
                    else
                    {
                        writer.WriteLine($"+{line.Content}");
                    }
                }
                else if (line.Type == TextDiffLineType.Deleted)
                {
                    if (isOldSide)
                    {
                        writer.WriteLine($"-{line.Content}");
                    }
                    else
                    {
                        if (!revert)
                        {
                            writer.WriteLine($" {line.Content}");
                        }
                        else
                        {
                            selection.IgnoredDeletes++;
                        }
                    }
                }
            }

            writer.WriteLine($" {tail}");
            writer.Flush();
        }

        private bool ProcessIndicatorForPatch(StreamWriter writer, TextDiffLine indicator, int idx, int start, int end, int ignoreRemoves, int ignoreAdds, bool revert, bool tailed)
        {
            var match = REG_INDICATOR().Match(indicator.Content);
            var oldStart = int.Parse(match.Groups[1].Value);
            var newStart = int.Parse(match.Groups[2].Value) + ignoreRemoves - ignoreAdds;
            var oldCount = 0;
            var newCount = 0;
            for (int i = idx + 1; i < end; i++)
            {
                var test = Lines[i];
                if (test.Type == TextDiffLineType.Indicator)
                    break;

                if (test.Type == TextDiffLineType.Normal)
                {
                    oldCount++;
                    newCount++;
                }
                else if (test.Type == TextDiffLineType.Added)
                {
                    if (i < start - 1)
                    {
                        if (revert)
                        {
                            newCount++;
                            oldCount++;
                        }
                    }
                    else
                    {
                        newCount++;
                    }

                    if (i == end - 1 && tailed)
                    {
                        newCount++;
                        oldCount++;
                    }
                }
                else if (test.Type == TextDiffLineType.Deleted)
                {
                    if (i < start - 1)
                    {
                        if (!revert)
                        {
                            newCount++;
                            oldCount++;
                        }
                    }
                    else
                    {
                        oldCount++;
                    }

                    if (i == end - 1 && tailed)
                    {
                        newCount++;
                        oldCount++;
                    }
                }
            }

            if (oldCount == 0 && newCount == 0)
                return false;

            writer.WriteLine($"@@ -{oldStart},{oldCount} +{newStart},{newCount} @@");
            return true;
        }

        private bool ProcessIndicatorForPatchSingleSide(StreamWriter writer, TextDiffLine indicator, int idx, int start, int end, int ignoreRemoves, int ignoreAdds, bool revert, bool isOldSide, bool tailed)
        {
            var match = REG_INDICATOR().Match(indicator.Content);
            var oldStart = int.Parse(match.Groups[1].Value);
            var newStart = int.Parse(match.Groups[2].Value) + ignoreRemoves - ignoreAdds;
            var oldCount = 0;
            var newCount = 0;
            for (int i = idx + 1; i < end; i++)
            {
                var test = Lines[i];
                if (test.Type == TextDiffLineType.Indicator)
                    break;

                if (test.Type == TextDiffLineType.Normal)
                {
                    oldCount++;
                    newCount++;
                }
                else if (test.Type == TextDiffLineType.Added)
                {
                    if (i < start - 1 || isOldSide)
                    {
                        if (revert)
                        {
                            newCount++;
                            oldCount++;
                        }
                    }
                    else
                    {
                        newCount++;
                    }

                    if (i == end - 1 && tailed)
                    {
                        newCount++;
                        oldCount++;
                    }
                }
                else if (test.Type == TextDiffLineType.Deleted)
                {
                    if (i < start - 1)
                    {
                        if (!revert)
                        {
                            newCount++;
                            oldCount++;
                        }
                    }
                    else
                    {
                        if (isOldSide)
                        {
                            oldCount++;
                        }
                        else
                        {
                            if (!revert)
                            {
                                newCount++;
                                oldCount++;
                            }
                        }
                    }

                    if (i == end - 1 && tailed)
                    {
                        newCount++;
                        oldCount++;
                    }
                }
            }

            if (oldCount == 0 && newCount == 0)
                return false;

            writer.WriteLine($"@@ -{oldStart},{oldCount} +{newStart},{newCount} @@");
            return true;
        }

        [GeneratedRegex(@"^@@ \-(\d+),?\d* \+(\d+),?\d* @@")]
        private static partial Regex REG_INDICATOR();
    }

    public class LFSDiff
    {
        public LFSObject Old { get; set; } = new LFSObject();
        public LFSObject New { get; set; } = new LFSObject();
    }

    public class BinaryDiff
    {
        public long OldSize { get; set; } = 0;
        public long NewSize { get; set; } = 0;
    }

    public class ImageDiff
    {
        public Bitmap Old { get; set; } = null;
        public Bitmap New { get; set; } = null;

        public long OldFileSize { get; set; } = 0;
        public long NewFileSize { get; set; } = 0;

        public string OldImageSize => Old != null ? $"{Old.PixelSize.Width} x {Old.PixelSize.Height}" : "0 x 0";
        public string NewImageSize => New != null ? $"{New.PixelSize.Width} x {New.PixelSize.Height}" : "0 x 0";
    }

    public class NoOrEOLChange;

    public class SubmoduleDiff
    {
        public RevisionSubmodule Old { get; set; } = null;
        public RevisionSubmodule New { get; set; } = null;
    }

    public class DiffResult
    {
        public bool IsBinary { get; set; } = false;
        public bool IsLFS { get; set; } = false;
        public string OldHash { get; set; } = string.Empty;
        public string NewHash { get; set; } = string.Empty;
        public string OldMode { get; set; } = string.Empty;
        public string NewMode { get; set; } = string.Empty;
        public TextDiff TextDiff { get; set; } = null;
        public LFSDiff LFSDiff { get; set; } = null;

        public string FileModeChange
        {
            get
            {
                if (string.IsNullOrEmpty(OldMode) && string.IsNullOrEmpty(NewMode))
                    return string.Empty;

                var oldDisplay = string.IsNullOrEmpty(OldMode) ? "0" : OldMode;
                var newDisplay = string.IsNullOrEmpty(NewMode) ? "0" : NewMode;

                return $"{oldDisplay} → {newDisplay}";
            }
        }
    }
}
