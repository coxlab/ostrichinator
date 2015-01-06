function res = vl_simplenn_fast(net, x, dzdy, res, varargin)

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.conserveMemory = true ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1)) ;
end
res(1).x = x ;

for i=1:n
  l = net.layers{i} ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
    case 'normalize'
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
    case 'relu'
      res(i+1).x = vl_nnrelu(res(i).x) ;
    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
    case 'dropout'
      if opts.disableDropout
        res(i+1).x = res(i).x ;
      elseif opts.freezeDropout
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  if opts.conserveMemory & ~doder %% & i < numel(net.layers) - 1
    % TODO: forget unnecesary intermediate computations even when
    % derivatives are required
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
end

if doder
  if isa(dzdy, 'function_handle')
    res(n+1).dzdx = dzdy(res(end).x(:));
    res(n+1).dzdx = reshape(res(n+1).dzdx, [1 1 size(res(n+1).dzdx)]) ;
  else
    res(n+1).dzdx = dzdy ;
  end
  for i=n:-1:1
    l = net.layers{i} ;
    
    if (size(res(i+1).dzdx,4) > size(res(i).x,4))
        res(i).x = repmat(res(i).x(:,:,:,1), [1 1 1 size(res(i+1).dzdx,4)]);
    end
    
    switch l.type
      case 'conv'
        [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
            vl_nnconv(res(i).x, l.filters, l.biases, ...
                      res(i+1).dzdx, ...
                      'pad', l.pad, 'stride', l.stride, 'NoDerFilters', 'NoDerBiases') ;
      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
          'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
      case 'normalize'
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'softmaxloss'
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'relu'
        res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
      case 'noffset'
        res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
      case 'dropout'
        if opts.disableDropout
          res(i).dzdx = res(i+1).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux) ;
        end
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
      res(i).x = [];
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
  end
end
