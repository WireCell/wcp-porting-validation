#!/usr/bin/perl
#
# Load every event's MABC zips from work/evt<ID>/ into data/<bee_idx>/.
# Each MABC job runs with initial_index=0, so all of its zips ship JSON files
# under data/0/0-*.json — concatenating raw would collide.  This script assigns
# a unique bee index per event and rewrites both the staging directory and the
# filename prefix on extract.
#
# Inputs:   work/evt<ID>/mabc-*.zip       (from ./run_clus_evt.sh all)
# Output:   data/<bee_idx>/<bee_idx>-*.json   (one bee_idx per event)
#
# Events appear in numerical evt-ID order: bee 0 = evt2, bee 1 = evt9, ...
# Skips work/evt<ID>_<tag>/ selection-tagged dirs.

use strict;
use warnings;
use File::Path qw(make_path remove_tree);
use File::Copy qw(move);
use File::Basename;

my $work_dir = 'work';
my $data_dir = 'data';

remove_tree($data_dir) if -d $data_dir;
make_path($data_dir);

my @evt_dirs = sort {
    my ($na) = ($a =~ /evt(\d+)$/);
    my ($nb) = ($b =~ /evt(\d+)$/);
    ($na // 0) <=> ($nb // 0);
} grep { -d $_ && m{/evt\d+$} } glob("$work_dir/evt*");

die "No work/evt<ID>/ directories found — run ./run_clus_evt.sh all first.\n"
    unless @evt_dirs;

my $bee_idx = 0;
for my $evt_dir (@evt_dirs) {
    my @zips = glob("$evt_dir/mabc-*.zip");
    if (!@zips) {
        warn "[skip] $evt_dir: no mabc-*.zip\n";
        next;
    }

    my $dst   = "$data_dir/$bee_idx";
    my $stage = "$dst/.stage";
    make_path($stage);

    for my $zip (@zips) {
        system('unzip', '-q', '-o', $zip, '-d', $stage) == 0
            or die "unzip failed for $zip\n";
    }

    for my $src_subdir (glob("$stage/data/*")) {
        next unless -d $src_subdir;
        my $src_idx = basename($src_subdir);
        for my $src_file (glob("$src_subdir/*")) {
            my $base = basename($src_file);
            (my $new_base = $base) =~ s/^\Q$src_idx\E-/$bee_idx-/;
            move($src_file, "$dst/$new_base")
                or die "move failed: $src_file → $dst/$new_base: $!\n";
        }
    }
    remove_tree($stage);

    my ($evtid) = ($evt_dir =~ /evt(\d+)$/);
    my @files = glob("$dst/*");
    printf "evt %-4s → data/%d  (%d files)\n", $evtid, $bee_idx, scalar @files;
    $bee_idx++;
}

print "Loaded $bee_idx event(s) into $data_dir/\n";
